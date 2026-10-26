# Tuning SonicMoE's expert GEMMs (branch `bahia-tuning`)

Companion to `sonicmoe_kernel_brief.md`. The plan behind this work, with the code review that motivates
it, is summarized in the brief's repo-side notes; this file says what was built and how to run it.

## What is in this branch

- `sonicmoe/tuning.py`: per-GEMM knobs for the six expert GEMMs, installed at import with bit-identical
  defaults. Roles: `up_act`, `down`, `dact`, `dw2`, `dx`, `dw1`.
- `sonicmoe/functional/__init__.py`, `backward.py`: the call sites read the knobs (`dynamic_scheduler`,
  `batch_idx_permute` for the varlen-K weight gradients) and carry optional profiler ranges. The duplicate
  `ds` scatter in `_down_projection_backward_act` is gone (same values were written twice).
- `tests/tuning_test.py`: spec parsing, role inference, autotune-space filtering, and GPU bit-identity of the
  default path against the pristine upstream path (`SONICMOE_TUNING_HOOKS=0`).
- In bahiamodel: `bahia/experiments/bench_sonic.py` (the harness), `bahia/experiments/sonic_sweeps.json`
  (the SM90 shortlist), `--sonic_tuning` on `bahia/training/train.py`, and `SONIC_MOE_REPO` /
  `SONIC_MOE_COMMIT` environment overrides for the Modal image in `cloud/shared.py`.

## Knobs

`SONICMOE_TUNING` (inline JSON or a path), or `sonicmoe.tuning.set_overrides(...)` before the first call:

```json
{"dw":  {"configs": ["128x320 coop c2x1", "256x160x128 coop c1x1"], "lpt_permute": true},
 "dw1": {"dynamic_scheduler": true},
 "all": {"dynamic_scheduler": false}}
```

Explicit role > `dw` (both weight gradients) > `all`, merged field by field.

| knob | what it does | crosses torch.compile | math |
|---|---|---|---|
| `configs` | the autotuner's space for that role becomes exactly this list (shapes QuACK lacks are added) | yes: the autotuner runs inside QuACK's custom-op body | bit-exact (ds may move in the last bit when `dact`'s tile_n changes) |
| `dynamic_scheduler` | QuACK's atomic-counter persistent scheduler instead of static round-robin (SM90) | yes, via the op schemas | bit-exact |
| `lpt_permute` | process experts longest-first (`batch_idx_permute`), dw roles only | yes | bit-exact |

Config spec: `<tile_m>x<tile_n>[x<tile_k>] [coop|pingpong] [c<cluster_m>x<cluster_n>] [swz<n>] [swap]`.
Changing overrides needs a fresh process: a tuned shape keeps its winner for the life of the process, and
QuACK's on-disk cache keys on the config list, so a changed space never replays a stale winner.

`SONICMOE_PROFILE_RANGES=1` wraps every GEMM and the two combine kernels in
`torch.profiler.record_function("sonicmoe/<role>")`. `sonicmoe.tuning.winners()` reports, per role, the
autotuner's chosen config, the scheduler, and QuACK's own L2-cold timing of every candidate it benchmarked.

## Why these knobs (from the code review)

- Every backward GEMM hard-coded `dynamic_scheduler=False`; QuACK resolves the scheduler as
  `dynamic_scheduler or config.is_dynamic_persistent` and every SM90 config has the latter False, so the
  backward ran the static scheduler. Confirmed.
- The weight-gradient GEMMs (per expert M=1280, N=640, varlen K, gathered A) tuned to a (256,160)
  cooperative tile with cluster (2,1), the only N=160 cooperative tile in QuACK's SM90 space. M=1280 is five
  256-row tiles, rounded up to three clusters of two: one CTA in six computes an out-of-range tile for the
  whole K loop. Cluster (1,1) and the 128-row cooperative tiles were not in the space.
- Those GEMMs re-gather A once per N-tile and are closer to the L2-bandwidth floor than to the MMA floor,
  so a wider tile_n (320, two N-tiles) is the structural candidate.
- Profiler kernel names do not distinguish configs or layouts (on the laptop all four default-epilogue GEMMs
  collapse onto one name), which is why the kernel map comes from ranges, not names.

## H100 run order

1. `python -m bahia.experiments.bench_sonic --experts 128,64` with no overrides: confirms the kernel map
   (which role owns the 45.8 ms kernel of the r3smoke19 profile) and the per-role TFLOPS baseline.
2. `--overrides bahia/experiments/sonic_sweeps.json:dw1_dynamic`: does gather_A + the dynamic scheduler
   work on SM90? QuACK's own space excludes that pairing for an unstated reason; parity must be bit-exact.
   If it hangs or fails, drop `dynamic_scheduler` for the dw roles and keep `lpt_permute`.
3. `--sweep bahia/experiments/sonic_sweeps.json`: every setting in its own process, on balanced, 1.5x and
   10x routing. Pick per-role winners within 2% of the best on balanced and 1.5x that do not regress at 10x.
   The `timings_ms` tables give each candidate's L2-cold time without a separate pin-and-rerun per config.
4. A 30-step 8xH100 smoke with `--sonic_tuning '<winners json>'`: compare `fwd_bwd` and step time against
   r3c's ~615 ms; the target is <= 592 ms.

## What the laptop (SM120) can and cannot tell you

It runs the same code through `GemmSm120` (`mma.sync`, CLC persistence, no clusters), so it validates the
plumbing, role inference, parity, range attribution and `lpt_permute`. It cannot measure any SM90
configuration; the sweep list is for the H100.

## Verified on the laptop (RTX 5050, SM120, 2026-10-25)

- `pytest tests/tuning_test.py`: 14 passed. The default path is bit-identical to the pristine upstream path
  on y, dx, dW1, dW2; `lpt_permute` and a single-config pin are bit-exact against it; every role tunes and
  every range carries device time.
- `pytest tests/moe_test.py` subset (SwiGLU and ReLU^2, with and without bias, 128 experts top-8 and 64 top-4):
  4 passed on the modified call sites.
- `bench_sonic.py --tokens 4096 --experts 128 --routing balanced,skew10 --torch_ref`: per-role times, the
  kernel -> role map, QuACK's candidate tables, and relative error vs the torch backend of 4e-3 to 6e-3 on
  y, dh, dW1, dW2 (bf16; < 0.03 is a match). The autotuner's L2-cold candidate times run above the warm
  per-role times measured in place, as expected.
