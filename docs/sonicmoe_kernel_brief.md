# SonicMoE expert-kernel work: brief for the laptop agent

## Goal

r3c (`m11_r3_moe_e128k8_b524k`, 8×H100) reached CORE 0.2577, past the GPT-2 target of 0.2565, in 6,177 s of
training time on nanochat's measure. nanochat's record is 5,940 s. Beating it needs about **4% off the step**:
from ~615 ms to ≤ ~592 ms per 524k-token step, without changing the model's math.

The routed-expert GEMMs are the largest block of the step and run at about half of the H100's bf16 peak.
They are the main place left to find that 4%.

## What the experts compute

Use these R3 shapes for every benchmark:

| | |
|---|---|
| Model width d | 1280 |
| Expert hidden | 640 (the dense ReLU² hidden 5120 / top-k 8) |
| Experts / top-k | 128 / 8 (also check 64 / 8, the earlier R3 shape) |
| Activation | ReLU² (`ActivationType.RELU_SQ`), not SwiGLU |
| Tokens per micro-batch per GPU | 16 × 2048 = 32,768, so 262,144 routed rows |
| Rows per expert | ~2,048 on average at 128 experts; settled load is ~1.5× max/mean, early training up to ~12× |
| Layers | 20 MoE layers, so every per-step figure below is a sum over 20 calls |
| Dtypes | bf16 weights, activations and weight gradients (the fp32 masters live in the optimizer) |

Our call is `bahia/model/moe.py::MoEMLP._sonic`. It calls
`sonicmoe.moe_general_routing_inputs(x, weights, token_idx, expert_idx, w1, None, w2, None, E, stream, RELU_SQ)`:
- routing is computed outside SonicMoE (sigmoid scores, aux-loss-free bias, top-8);
- `token_idx` is sorted;
- `w1` is `(hidden, d, E)` and `w2` is `(d, hidden, E)`, permuted views of `(E, hidden, d)` / `(E, d, hidden)` parameters.

Pinned versions (`cloud/shared.py`): sonic-moe at commit `7396f3e604827d8186c2e16e64b28ee33d3defd0`,
`quack-kernels[cu13]==0.6.5`, torch 2.14.1. Use the same versions on the laptop, or changes won't port.

## Where the time goes (H100, rank 0, one 262k-token micro-batch, 20 layers)

From the r3smoke19 profile (64 experts). 128 experts has the same FLOPs and total rows:

| Kernel (QuACK / SonicMoE) | ms per step | Probably |
|---|---|---|
| `quackgemm_default_epi`, ThrLayout 128×2, tile 64×160×16 | 45.8 | the largest backward GEMM(s) |
| `quack…GemmMod_dact_epi` | 17.4 | down-projection backward with the ReLU² derivative fused |
| `quack…GemmMod_linear_act_epi` | 15.7 | forward up-projection with ReLU² in the epilogue |
| `quackgemm_default_epi`, tile 64×128×16 | 13.1 | forward down-projection or a dX GEMM |
| `quackgemm_default_epi`, ThrLayout 128×1, tile 64×160×16 | 12.0 | another backward GEMM (weight gradient?) |
| `token_gather_sum_kernel` | 10.1 | the weighted combine; ~0.5 ms per layer, near its bandwidth floor |
| **Expert GEMMs total** | **~104** | |

The "probably" column is a guess from kernel names. Confirm the mapping first: it is task 1 below.

Each layer's experts do 2.58 TFLOP of forward and backward per micro-batch (6 × 262,144 × 1280 × 640 × 2).
Over 20 layers in ~104 ms that is ~495 TFLOPS, about 50% of the bf16 peak. The SonicMoE paper reports
49–63% of peak on a similar fine-grained configuration. **Realistic headroom is 10–25% of these kernels:
5–15 ms per micro-batch, 10–30 ms per 524k step**, which covers the 4% (~23 ms).

The combine kernel and the routing preparation are already near their floors. Leave them alone.

## Candidate levers, in order

1. **Tile, cluster and pipeline configuration per GEMM.** Our expert GEMMs are narrow:
   - forward up: N = 640, K = 1280;
   - forward down: N = 1280, K = 640;
   - weight gradients: K is the rows per expert, about 2,048 and variable.

   SonicMoE's defaults were tuned on other shapes. The paper notes that the best M-tile depends on the shape,
   and that "a larger tile may be preferred for small d or n". Find where each of the GEMMs (forward up with
   gather prologue and ReLU² epilogue, forward down, the backward dact/dX GEMMs, the varlen-K weight-gradient
   GEMMs) picks its tile, cluster, stages and scheduler. Expose those as overrides: kwargs or env vars, defaults
   unchanged.
2. **Persistent / dynamic tile scheduler in backward.** A team review said SonicMoE leaves its dynamic scheduler
   off in backward. That is **unverified**. Check the code, and if it's true, expose it as an option. With
   variable rows per expert, especially early in training, a persistent scheduler should help the varlen GEMMs.
3. **Token rounding / padding.** SonicMoE pads each expert's rows to the tile size. At ~2,048 rows per expert the
   waste is small (~1.5%). Low priority. Don't change routing semantics.
4. **Anything that removes a pass over the 262k×1280 gathered activations** in backward without changing results.

Out of scope: FP8 (SonicMoE has no FP8 path, and our widths are too small for it to pay), and changes to the
routing math.

## What carries over from the laptop's GPU (SM120) and what doesn't

The RTX 50-series laptop GPU is consumer Blackwell (SM120). The servers are Hopper (SM90, H100) and datacenter
Blackwell (SM100, B200). Their tensor-core paths differ:
- the H100 kernels in the profile are QuACK's SM90 kernels (`GemmDefaultSm90`, warpgroup MMA);
- SM100 uses tcgen05 and tensor memory;
- SM120 has neither of those MMA paths (check the QuACK source for exactly what it uses on SM120).

So whatever SonicMoE runs on the laptop is a different kernel path from the one on the H100.

**Transfers:**
- correctness work and the parity harness;
- the plumbing that exposes per-GEMM tile, cluster, stage and scheduler choices (the knobs themselves);
- the map from each profiler kernel name to the GEMM it computes;
- algorithmic changes that remove passes or fuse work, if written for the SM90 path;
- the benchmark script, if it runs unchanged on the H100.

**Doesn't transfer:**
- which tile, cluster or stage count is fastest;
- occupancy or register tuning;
- any absolute timing.

Tuning has to happen on an H100. The laptop builds the instruments; the H100 runs the sweep.

## Deliverables for the laptop

1. **The kernel map.** For each H100 kernel in the table above, name the GEMM it computes (operands and which
   pass), the file and function that launches it, and the tile/cluster/stages/scheduler it uses at our shapes.
2. **Override knobs.** A minimal patch to sonic-moe (or QuACK) at the pinned versions, or a wrapper, that lets a
   caller set those per GEMM. Default behaviour must be bit-identical.
3. **`bahia/experiments/bench_sonic.py`.** A standalone script that:
   - builds one `MoEMLP(1280, E, 8, 640, backend="sonic")` layer;
   - feeds it bf16 inputs with realistic routing: balanced, plus a 1.5× and a 10× skewed case;
   - runs forward and backward;
   - checks parity against the current defaults (bit-exact for scheduler/tile changes that shouldn't change math,
     otherwise within bf16 tolerance) and against the torch backend;
   - reports per-kernel CUDA time from `torch.profiler` and TFLOPS;
   - takes a list of knob settings to sweep;
   - runs unchanged on an H100.

   On the laptop's 8 GB, use 4,096–8,192 tokens instead of 32,768. On the H100, use the full 32,768.
4. **A shortlist of SM90 configurations to sweep**, from reading QuACK's supported tile and cluster shapes for each GEMM.

I'll add a Modal entry point that runs `bench_sonic.py` on one H100 with the sweep list and reports the best
configuration per GEMM. A 30-step 8×H100 smoke run then confirms the step-time change.
