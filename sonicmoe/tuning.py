# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""Per-GEMM tuning knobs for SonicMoE's six expert GEMMs. Defaults are bit-identical to upstream.

Roles (one per grouped GEMM a layer launches):

    up_act  forward up-projection, gather on A, activation epilogue (QuACK ``gemm_act``)
    down    forward down-projection                                  (QuACK ``gemm``)
    dact    backward dH kernel, gather on A, dact epilogue           (QuACK ``gemm_dact``)
    dw2     backward dW2, varlen-K, gather dout^T along K            (QuACK ``gemm``)
    dx      backward dX                                               (QuACK ``gemm``)
    dw1     backward dW1, varlen-K, gather x^T along K               (QuACK ``gemm``)

Knobs per role, from the env var ``SONICMOE_TUNING`` (inline JSON or a path to a JSON file) or
``set_overrides(...)`` before the first MoE call:

    {"dw1": {"configs": ["128x320 coop c2x1", "256x160x128 coop c1x1"],
             "dynamic_scheduler": true, "lpt_permute": true},
     "dw":  {...},        # alias for dw1 and dw2
     "all": {...}}        # applies to every role not listed explicitly

* ``configs``: the autotuner's config space for that role becomes exactly this list. A spec is
  ``<tile_m>x<tile_n>[x<tile_k>] [coop|pingpong] [c<cluster_m>x<cluster_n>] [swz<n>] [swap]``.
  Shapes missing from QuACK's space are added to it (for every role; the hook removes them again for
  roles without an override, so those roles keep QuACK's default space and winner). Under
  ``torch.compile`` an explicit config cannot cross QuACK's custom-op boundary; steering the
  autotuner's space, which runs eagerly inside the op body, is the one mechanism that works there.
* ``dynamic_scheduler``: QuACK's atomic-counter persistent scheduler (SM90) instead of the static
  round-robin one. Upstream SonicMoE hard-codes False on every backward GEMM.
* ``lpt_permute`` (dw roles): process experts longest-first (``batch_idx_permute`` = argsort of the
  expert row counts, descending). Order only; the math is bit-identical.

``SONICMOE_PROFILE_RANGES=1`` wraps each GEMM in ``torch.profiler.record_function("sonicmoe/<role>")``
so a profile attributes kernels to roles. ``SONICMOE_TUNING_HOOKS=0`` installs nothing at all (the
pristine upstream path, for bit-identity checks).

Tile, cluster, scheduler and expert order never change the K-order of the fp32 accumulation, so every
GEMM output is bit-exact across them. Exception: ``dact``'s ds is a sum of per-N-tile partials, so a
different ``dact`` tile_n can move ds (the router gradient) in the last bit.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import warnings
from dataclasses import dataclass, replace
from typing import Optional

import torch

import quack.autotuner
import quack.gemm_config as _gc
import quack.gemm_interface as _gi
import quack.gemm_runtime.autotune as _qa
from quack.gemm_config import GemmConfig


ROLES = ("up_act", "down", "dact", "dw2", "dx", "dw1")
DW_ROLES = ("dw1", "dw2")
# range-only names (no knobs): the two Triton combine kernels and the routing metadata
RANGE_NAMES = ROLES + ("combine", "dx_reduce", "metadata")

ENV_OVERRIDES = "SONICMOE_TUNING"
ENV_HOOKS = "SONICMOE_TUNING_HOOKS"
ENV_RANGES = "SONICMOE_PROFILE_RANGES"


# ---------------------------------------------------------------------------------------------
# config specs
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigSpec:
    """An arch-independent GEMM configuration; ``to_gemm_config`` binds it to a device capability."""

    tile_m: int
    tile_n: int
    tile_k: Optional[int] = None
    pingpong: bool = False
    cluster_m: int = 1
    cluster_n: int = 1
    swizzle: int = 8
    swap_ab: bool = False

    _SPEC_RE = re.compile(r"^(\d+)x(\d+)(?:x(\d+))?$")

    @classmethod
    def parse(cls, text: str) -> "ConfigSpec":
        tokens = text.replace(",", " ").split()
        if not tokens:
            raise ValueError("empty config spec")
        m = cls._SPEC_RE.match(tokens[0])
        if m is None:
            raise ValueError(f"config spec {text!r}: first token must be <tile_m>x<tile_n>[x<tile_k>]")
        tile_m, tile_n = int(m.group(1)), int(m.group(2))
        tile_k = int(m.group(3)) if m.group(3) else None
        pingpong, cluster_m, cluster_n, swizzle, swap_ab = False, 1, 1, 8, False
        for tok in tokens[1:]:
            t = tok.lower()
            if t in ("coop", "cooperative"):
                pingpong = False
            elif t in ("pingpong", "pp"):
                pingpong = True
            elif t.startswith("c") and "x" in t and t[1:].replace("x", "").isdigit():
                cluster_m, cluster_n = (int(v) for v in t[1:].split("x"))
            elif t.startswith("swz") and t[3:].isdigit():
                swizzle = int(t[3:])
            elif t in ("swap", "swap_ab"):
                swap_ab = True
            else:
                raise ValueError(f"config spec {text!r}: unknown token {tok!r}")
        return cls(tile_m, tile_n, tile_k, pingpong, cluster_m, cluster_n, swizzle, swap_ab)

    @classmethod
    def from_gemm_config(cls, c: GemmConfig) -> "ConfigSpec":
        return cls(c.tile_m, c.tile_n, c.tile_k, c.pingpong, c.cluster_m, c.cluster_n, c.max_swizzle_size, c.swap_ab)

    def to_gemm_config(self, device_capacity: int) -> GemmConfig:
        """The GemmConfig QuACK's space would hold for this spec on this arch (same defaults as
        ``quack.gemm_config._get_sm90_configs`` / ``_get_sm120_configs``, so equality works)."""
        return GemmConfig(
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            pingpong=self.pingpong,
            cluster_m=self.cluster_m,
            cluster_n=self.cluster_n,
            swap_ab=self.swap_ab,
            max_swizzle_size=self.swizzle,
            device_capacity=device_capacity,
            # SM90 has no CLC; SM120's dynamic persistence is CLC and on by default in QuACK's space
            is_dynamic_persistent=(device_capacity == 12),
            use_tma_gather=False,
        )

    def __str__(self) -> str:
        shape = f"{self.tile_m}x{self.tile_n}" + (f"x{self.tile_k}" if self.tile_k else "")
        parts = [shape, "pingpong" if self.pingpong else "coop", f"c{self.cluster_m}x{self.cluster_n}"]
        if self.swizzle != 8:
            parts.append(f"swz{self.swizzle}")
        if self.swap_ab:
            parts.append("swap")
        return " ".join(parts)


def describe(config: GemmConfig) -> str:
    return str(ConfigSpec.from_gemm_config(config))


@dataclass(frozen=True)
class RoleOverride:
    configs: Optional[tuple[ConfigSpec, ...]] = None
    dynamic_scheduler: Optional[bool] = None
    lpt_permute: Optional[bool] = None

    @classmethod
    def parse(cls, d: dict) -> "RoleOverride":
        unknown = set(d) - {"configs", "dynamic_scheduler", "lpt_permute"}
        if unknown:
            raise ValueError(f"unknown override keys {sorted(unknown)}")
        cfgs = d.get("configs")
        return cls(
            configs=tuple(ConfigSpec.parse(s) if isinstance(s, str) else s for s in cfgs) if cfgs else None,
            dynamic_scheduler=d.get("dynamic_scheduler"),
            lpt_permute=d.get("lpt_permute"),
        )


def _merge(base: RoleOverride, top: RoleOverride) -> RoleOverride:
    """Field-wise: a field set on ``top`` wins, an unset one falls through to ``base``."""
    return RoleOverride(
        configs=top.configs if top.configs is not None else base.configs,
        dynamic_scheduler=top.dynamic_scheduler if top.dynamic_scheduler is not None else base.dynamic_scheduler,
        lpt_permute=top.lpt_permute if top.lpt_permute is not None else base.lpt_permute,
    )


def _expand_roles(mapping: dict) -> dict[str, RoleOverride]:
    """{"all": ..., "dw": ..., "<role>": ...} -> one RoleOverride per concrete role, merged field-wise
    with the explicit role over ``dw`` over ``all``."""
    parsed = {}
    for key, value in mapping.items():
        if key not in ROLES and key not in ("dw", "all"):
            raise ValueError(f"unknown role {key!r}; roles are {ROLES} plus the aliases 'dw' and 'all'")
        parsed[key] = value if isinstance(value, RoleOverride) else RoleOverride.parse(dict(value))
    out: dict[str, RoleOverride] = {}
    for role in ROLES:
        ov = RoleOverride()
        for key in ("all", "dw", role):
            if key in parsed and (key != "dw" or role in DW_ROLES):
                ov = _merge(ov, parsed[key])
        if ov != RoleOverride():
            out[role] = ov
    return out


# SM90 shapes QuACK's space omits that fit N=640 / M=1280 expert GEMMs (see docs/sonicmoe_kernel_brief.md).
# All satisfy GemmSm90.__init__'s shape rules; accumulator registers stay under the cooperative budget.
def _default_sm90_extras() -> list[ConfigSpec]:
    specs: list[ConfigSpec] = []
    for n in (128, 160, 192):
        specs.append(ConfigSpec(256, n, None, False, 1, 1))                        # coop 256 without a cluster
        for cm, cn in ((2, 1), (1, 1), (1, 2)):
            specs.append(ConfigSpec(128, n, None, False, cm, cn))                  # coop 128 (2 warpgroups x 64 rows)
        specs.append(ConfigSpec(128, n, None, True, 1, 1))                         # pingpong without a cluster
    # (128, 320) is NOT legal on SM90: one warpgroup's wgmma N is capped at 256 (a compile-time abort).
    # (320, 160), the four-tile M option, aborts inside CuTeDSL for these layouts (see sm90_spec_problem).
    for cm, cn in ((2, 1), (1, 1)):                                                # deeper K per pipeline stage
        specs.append(ConfigSpec(128, 160, 128, False, cm, cn))                     # (256-row tiles leave one stage)
        specs.append(ConfigSpec(128, 160, 128, True, cm, cn))
    return specs


# ---------------------------------------------------------------------------------------------
# SM90 shape validity (a bad shape is a CuTeDSL compile error that aborts the process, not an inf timing)
# ---------------------------------------------------------------------------------------------
SM90_SMEM_BYTES = 227 * 1024
_SM90_EPI_RESERVE_BYTES = 48 * 1024   # epilogue stages, barriers and alignment pads, roughly


def _sm90_atom_layout(spec: ConfigSpec) -> tuple[int, int]:
    """The warpgroup layout GemmSm90.__init__ derives from the tile (quack/gemm_sm90.py)."""
    if spec.pingpong:
        return 1, 1
    if spec.tile_m == 320:
        return 1, 2
    if spec.tile_m == 192:
        return (3, 1) if spec.tile_n <= 128 else (1, 2)
    return (spec.tile_m // 64 if spec.tile_m < 256 else 2), 1


def sm90_spec_problem(spec: ConfigSpec) -> Optional[str]:
    """Why QuACK's SM90 kernel cannot be built for this spec, or None. Mirrors GemmSm90.__init__'s tile rules
    and adds two it does not check up front: the per-warpgroup wgmma N must be <= 256 and a multiple of 8
    (tile_n 320 with one warpgroup along N fails there, inside the compiler), and the A/B pipeline needs at
    least two stages in the 227 KB of shared memory."""
    m, n, k = spec.tile_m, spec.tile_n, spec.tile_k or 64
    if k % 16:
        return f"tile_k {k} is not a multiple of 16"
    if spec.pingpong:
        if m not in (64, 128, 192):
            return f"pingpong needs tile_m 64, 128 or 192, got {m}"
        n_max = 256 if m == 64 else (208 if m == 128 else 128)
        if n % 16 or n > n_max:
            return f"pingpong tile_m {m} needs tile_n % 16 == 0 and <= {n_max}, got {n}"
    else:
        if m not in (64, 128, 192, 256, 320):
            return f"cooperative needs tile_m 64, 128, 192, 256 or 320, got {m}"
        if m in (192, 320):
            n_max = 256 if m == 192 else 160
            if n % 32 or n > n_max:
                return f"tile_m {m} needs tile_n % 32 == 0 and <= {n_max}, got {n}"
        elif not ((n % 16 == 0 and n <= 256) or (n % 32 == 0 and n <= 512)):
            return f"tile_n {n} must be a multiple of 16 up to 256 or of 32 up to 512"
    am, an = _sm90_atom_layout(spec)
    if an == 2:
        # Legal by GemmSm90.__init__'s rules, but the trace aborts the process (std::bad_variant_access) for every
        # SonicMoE operand layout tried: sonic_compile_check for sm_90a, 2026-10-26 (320x160 both clusters, 192x160,
        # 192x256; dw1, dw2 and dx). 192x128, which keeps one warpgroup along N, builds.
        return f"tile_m {m} with tile_n {n} splits N across two warpgroups, which aborts inside CuTeDSL"
    mma_n = n // an
    if mma_n > 256 or mma_n % 8:
        return f"wgmma N {mma_n} (tile_n {n} over {an} warpgroup(s) along N) must be <= 256 and a multiple of 8"
    acc_regs = m * n // (am * an * 128)
    if acc_regs > 232:
        return f"{acc_regs} accumulator registers per thread exceed the cooperative budget of 232"
    stage_bytes = (m + n) * k * 2
    if (SM90_SMEM_BYTES - _SM90_EPI_RESERVE_BYTES) // stage_bytes < 2:
        return f"{stage_bytes // 1024} KB per A/B stage leaves fewer than two pipeline stages"
    if spec.cluster_m not in (1, 2, 4) or spec.cluster_n not in (1, 2, 4) or spec.cluster_m * spec.cluster_n > 4:
        return "cluster dims must be 1, 2 or 4 with at most 4 CTAs"
    return None


_warned_specs: set = set()


def _spec_usable(spec: ConfigSpec, cap: int, where: str) -> bool:
    if cap != 9:
        return True   # SM120's constraints are not modelled; the laptop only validates plumbing
    why = sm90_spec_problem(spec)
    if why is None:
        return True
    if (spec, where) not in _warned_specs:
        _warned_specs.add((spec, where))
        warnings.warn(f"sonicmoe.tuning: skipping {spec} for {where} on SM90: {why}", stacklevel=3)
    return False


# ---------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------
_lock = threading.Lock()
_installed = False
_hooks_enabled = os.environ.get(ENV_HOOKS, "1") != "0"
_ranges_enabled = os.environ.get(ENV_RANGES, "0") == "1"
_overrides: dict[str, RoleOverride] = {}
_extra_specs: dict[int, set[ConfigSpec]] = {9: set(_default_sm90_extras()), 12: set()}
_default_space: dict[int, frozenset[GemmConfig]] = {}  # per arch: QuACK's space before our extras
_extra_configs: dict[int, frozenset[GemmConfig]] = {}  # per arch: the extras actually appended
_digest_role: dict[str, str] = {}                      # epilogue-mod semantic digest -> role
WINNERS: dict[str, dict] = {}                           # role -> {"config": GemmConfig, "dynamic_scheduler": bool, "timings": {...}}
_last_timings: dict[int, object] = {}                   # id(tuner) -> the configs_timings dict last recorded

_orig_get_sm90 = None
_orig_get_sm120 = None
_orig_prune_gemm = None
_orig_prune_mod = None
_orig_autotuner_call = None


def hooks_enabled() -> bool:
    return _hooks_enabled


def overrides() -> dict[str, RoleOverride]:
    return dict(_overrides)


def set_overrides(mapping: Optional[dict]) -> None:
    """Replace the overrides (``None`` or ``{}`` clears them). Call before the first MoE call: an
    autotuner that already picked a winner for a shape does not re-tune in this process."""
    global _overrides
    with _lock:
        _overrides = _expand_roles(mapping or {})
        for role_ov in _overrides.values():
            for spec in role_ov.configs or ():
                for cap in _extra_specs:
                    _extra_specs[cap].add(spec)
        if _installed:
            if _qa._MOD_TUNERS or _gi.gemm_tuned.cache:
                warnings.warn(
                    "sonicmoe.tuning.set_overrides after GEMMs were already tuned in this process: "
                    "cached winners are kept; start a fresh process to re-tune",
                    stacklevel=2,
                )
            _refresh_spaces()


def _load_env_overrides() -> Optional[dict]:
    raw = os.environ.get(ENV_OVERRIDES, "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        return json.loads(raw)
    with open(raw) as f:
        return json.load(f)


def dynamic_scheduler(role: str, default: bool = False) -> bool:
    ov = _overrides.get(role)
    return default if ov is None or ov.dynamic_scheduler is None else bool(ov.dynamic_scheduler)


def lpt_permute(role: str, expert_frequency_offset: torch.Tensor) -> Optional[torch.Tensor]:
    """``batch_idx_permute`` for a varlen-K GEMM: experts in descending row count, or None when off."""
    ov = _overrides.get(role)
    if ov is None or not ov.lpt_permute:
        return None
    num_experts = expert_frequency_offset.shape[0] - 1
    # QuACK's kernel signature assumes the permutation's length is a multiple of 4 (make_fake_scheduler_args)
    if num_experts % 4 != 0:
        return None
    counts = expert_frequency_offset[1:] - expert_frequency_offset[:-1]
    return torch.argsort(counts, descending=True, stable=True).to(torch.int32)


def profile_range(name: str):
    """A ``record_function("sonicmoe/<name>")`` range when SONICMOE_PROFILE_RANGES=1, else a no-op."""
    if _ranges_enabled:
        return torch.profiler.record_function(f"sonicmoe/{name}")
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------------------------
# role inference from what QuACK's hooks see
# ---------------------------------------------------------------------------------------------
def infer_role_gemm(args: dict) -> Optional[str]:
    """Role of a default-epilogue ``gemm`` call from its argument metadata. The stride rules follow
    from SonicMoE's weight views ((I, H, E) / (H, I, E), stride order (2, 0, 1)) and from
    ``torch.empty_like`` preserving them for dw1/dw2; see the brief's weight-format notes."""
    A, B, out = args.get("A"), args.get("B"), args.get("out")
    if A is None or B is None:
        return None
    if args.get("cu_seqlens_k") is not None:
        if out is None or out.ndim != 3:
            return None
        return "dw1" if out.stride(-2) == 1 else "dw2"   # dw1's D is M-major, dw2's is N-major
    if args.get("cu_seqlens_m") is not None and args.get("A_idx") is None:
        if B.ndim != 3:
            return None
        return "down" if B.stride(-2) == 1 else "dx"    # down: B K-major (w2^T view); dx: B N-major (w1 view)
    return None


def infer_role_mod(mod) -> Optional[str]:
    """Role of an epilogue-mod call: the generated epilogue function's name is structural."""
    name = getattr(getattr(mod, "fn", None), "__name__", "")
    if name == "linear_act_epi":
        return "up_act"
    if name == "dact_epi":
        return "dact"
    return None


def _role_from_tuner_call(tuner, args, kwargs) -> Optional[str]:
    if tuner is _gi.gemm_tuned:
        return infer_role_gemm({**dict(zip(tuner.arg_names, args)), **kwargs})
    digest = kwargs.get("mod_digest")
    if digest is not None and digest in _digest_role:
        return _digest_role[digest]
    if getattr(tuner.fn, "__name__", "").startswith("mod_"):
        # first sight of a mod tuner on a bucket hit: dact passes PreAct as C, up_act has no C
        return "dact" if kwargs.get("C") is not None else "up_act"
    return None


# ---------------------------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------------------------
def _with_extras(original, cap: int):
    def generator(epilogue=None, tune_coop=True):
        base = list(original(epilogue, tune_coop))
        base_set = frozenset(base)
        _default_space[cap] = base_set
        extras = []
        for spec in sorted(_extra_specs.get(cap, ()), key=str):
            c = spec.to_gemm_config(cap)
            if c in base_set or c in extras:
                continue
            if not _spec_usable(spec, cap, "the config space"):
                continue
            if epilogue == "gated" and (c.tile_n % 32 or c.tile_m == 192):
                continue
            if epilogue in ("lse", "gated") and c.swap_ab:
                continue
            extras.append(c)
        _extra_configs[cap] = frozenset(extras)
        return base + extras

    return generator


def _refresh_spaces() -> None:
    """Rebuild gemm_tuned's config list (it captured the space at decoration time)."""
    _gi.gemm_tuned.configs = [quack.autotuner.AutotuneConfig(config=c) for c in _gc.get_all_configs()]


def _filter_for_role(kept: list, role: Optional[str], cap: int, where: str) -> list:
    ov = _overrides.get(role) if role is not None else None
    if ov is not None and ov.configs:
        wanted = {spec.to_gemm_config(cap) for spec in ov.configs if _spec_usable(spec, cap, f"role {role}")}
        chosen = [conf for conf in kept if conf.kwargs["config"] in wanted]
        if not chosen:
            raise ValueError(
                f"sonicmoe.tuning: none of the configs listed for role {role!r} can run this {where} call "
                f"({[str(s) for s in ov.configs]}): the arch's shape rules (see sm90_spec_problem) or QuACK's "
                "structural checks (gather needs cluster_n=1, varlen/gather forbid swap_ab) removed them all"
            )
        return chosen
    extras = _extra_configs.get(cap)
    if not extras:
        return kept
    return [conf for conf in kept if conf.kwargs["config"] not in extras]   # exactly QuACK's default space


def _device_capacity(args: dict) -> int:
    from quack.cute_dsl_utils import get_device_capacity

    return get_device_capacity(args["A"].device)[0]


def _prune_gemm(configs, named_args, **kwargs):
    kept = _orig_prune_gemm(configs, named_args, **kwargs)
    merged = {**named_args, **kwargs}
    return _filter_for_role(kept, infer_role_gemm(merged), _device_capacity(merged), "gemm")


def _prune_mod(mod, transform_a, configs, named_args, **kwargs):
    kept = _orig_prune_mod(mod, transform_a, configs, named_args, **kwargs)
    merged = {**named_args, **kwargs}
    role = infer_role_mod(mod)
    if role is not None:
        _digest_role[mod.semantic_digest] = role
    return _filter_for_role(kept, role, _device_capacity(merged), f"{getattr(mod.fn, '__name__', 'mod')}")


@torch.compiler.disable
def _autotuner_call(self, *args, **kwargs):
    ret = _orig_autotuner_call(self, *args, **kwargs)
    best = getattr(self, "best_config", None)
    if best is not None and len(self.configs) > 1:
        role = _role_from_tuner_call(self, args, kwargs)
        if role is not None:
            cfg = best.kwargs.get("config")
            if cfg is not None:
                entry = WINNERS.setdefault(role, {})
                entry["config"] = cfg
                entry["dynamic_scheduler"] = bool(kwargs.get("dynamic_scheduler", False) or cfg.is_dynamic_persistent)
                # QuACK benchmarks every candidate (L2-cold CUDA graph) when it tunes a shape; keep that table.
                # configs_timings only changes on a fresh tune (in-memory or disk-cache miss).
                timings = self.__dict__.get("configs_timings")
                if timings is not None and timings is not _last_timings.get(id(self)):
                    _last_timings[id(self)] = timings
                    entry["timings"] = {
                        describe(c.kwargs["config"]): float(t[0] if isinstance(t, (list, tuple)) else t)
                        for c, t in timings.items()
                    }
    return ret


def winners() -> dict[str, dict]:
    """{role: {"config": "<spec>", "dynamic_scheduler": bool, "timings_ms": {"<spec>": ms, ...}}} for every role
    tuned so far in this process. ``timings_ms`` is QuACK's own L2-cold benchmark of every candidate in that
    role's space (ascending; inf = the candidate failed to compile or launch), present after a fresh tune."""
    out = {}
    for role, w in sorted(WINNERS.items()):
        entry = {"config": describe(w["config"]), "dynamic_scheduler": w["dynamic_scheduler"]}
        if "timings" in w:
            entry["timings_ms"] = dict(sorted(w["timings"].items(), key=lambda kv: kv[1]))
        out[role] = entry
    return out


def install() -> None:
    """Install the hooks once (a no-op with SONICMOE_TUNING_HOOKS=0). Called from sonicmoe.functional
    after its own QuACK patches, so the autotuner wrapper chains onto SonicMoE's M-bucketing one."""
    global _installed, _orig_get_sm90, _orig_get_sm120, _orig_prune_gemm, _orig_prune_mod, _orig_autotuner_call
    with _lock:
        if _installed or not _hooks_enabled:
            return
        env = _load_env_overrides()
        if env:
            global _overrides
            _overrides = _expand_roles(env)
            for role_ov in _overrides.values():
                for spec in role_ov.configs or ():
                    for cap in _extra_specs:
                        _extra_specs[cap].add(spec)
        _orig_get_sm90, _orig_get_sm120 = _gc._get_sm90_configs, _gc._get_sm120_configs
        _gc._get_sm90_configs = _with_extras(_orig_get_sm90, 9)
        _gc._get_sm120_configs = _with_extras(_orig_get_sm120, 12)
        _orig_prune_gemm = _gi.gemm_tuned.early_config_prune
        _gi.gemm_tuned.early_config_prune = _prune_gemm
        _orig_prune_mod = _qa._prune_for_mod
        _qa._prune_for_mod = _prune_mod
        _orig_autotuner_call = quack.autotuner.Autotuner.__call__
        quack.autotuner.Autotuner.__call__ = _autotuner_call
        _refresh_spaces()
        _installed = True
