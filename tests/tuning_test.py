# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""sonicmoe.tuning: the per-GEMM knobs leave the default path bit-identical, filter the autotune space per
role, infer roles from argument metadata, and attribute kernels to roles. The GPU cases run a small ReLU^2
layer (general routing, as bahia calls it) and compare against subprocesses with other settings."""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

from sonicmoe import tuning
from sonicmoe.tuning import ConfigSpec, RoleOverride

from .test_commons import TestCommons


# A small layer: T, H, I, E, K. E % 4 == 0 so lpt_permute is exercised.
SHAPE = (2048, 512, 256, 16, 4)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs one forward+backward of the layer in a fresh process (so SONICMOE_* env vars take effect at import),
# saving the outputs, and optionally the per-range device time under torch.profiler.
CHILD = r"""
import json, sys, torch
import sonicmoe
from sonicmoe import moe_general_routing_inputs, tuning
from sonicmoe.enums import ActivationType
T, H, I, E, K = json.loads(sys.argv[1]); out_path = sys.argv[2]; want_profile = sys.argv[3] == "1"
# Pin the Triton combine kernel: Triton re-times it per process and its fp32 reduction order follows BLOCK_K,
# which could otherwise flip a bf16 ulp of y / dx between two processes (the GEMM roles are bit-exact anyway).
import triton
from sonicmoe.functional.reduction_over_k_gather import token_gather_sum_kernel
token_gather_sum_kernel.configs = [triton.Config({"BLOCK_H": min(triton.next_power_of_2(H), 2048), "BLOCK_K": triton.next_power_of_2(K)},
                                                 num_warps=8, num_stages=4)]
torch.manual_seed(0)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
W1 = (torch.randn(E, I, H, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
W2 = (torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
scores, experts = torch.randn(T, E, device="cuda").topk(K)
scores = scores.softmax(-1)
tokens = torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(K)
dy = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
def run():
    y, _ = moe_general_routing_inputs(x, scores.flatten(), tokens, experts.flatten().int(), W1.permute(1, 2, 0), None,
                                      W2.permute(1, 2, 0), None, E, torch.cuda.current_stream().cuda_stream,
                                      ActivationType.RELU_SQ)
    return y, torch.autograd.grad(y, [x, W1, W2], dy)
y, (dx, dw1, dw2) = run()
torch.cuda.synchronize()
res = {"y": y.detach().cpu(), "dx": dx.cpu(), "dw1": dw1.cpu(), "dw2": dw2.cpu(), "winners": tuning.winners(),
       "hooks": tuning.hooks_enabled()}
if want_profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run(); torch.cuda.synchronize()
    res["range_ms"] = {r.key: r.device_time_total / 1e3 for r in prof.key_averages()
                       if r.key.startswith("sonicmoe/") and r.device_time_total > 0}
torch.save(res, out_path)
"""


def run_child(env_overrides: dict, want_profile: bool = False) -> dict:
    env = {**os.environ, **env_overrides}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        out_path = f.name
    cmd = [sys.executable, "-c", CHILD, json.dumps(SHAPE), out_path, "1" if want_profile else "0"]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-4000:]
    res = torch.load(out_path, weights_only=False)
    os.unlink(out_path)
    return res


def _views(E=16, H=512, I=256):
    """Tensors with SonicMoE's weight/gradient strides, for the role rules (CPU; only strides matter)."""
    w1 = torch.empty(E, I, H).permute(1, 2, 0)            # (I, H, E), as bahia passes it
    w2 = torch.empty(E, H, I).permute(1, 2, 0)            # (H, I, E)
    dw1 = torch.empty_like(w1)                            # preserves the strides
    dw2 = torch.empty_like(w2)
    return dict(
        down_B=w2.permute(2, 1, 0),                       # (E, I, H), K-major
        dx_B=w1.permute(2, 0, 1),                         # (E, I, H), N-major
        dw1_out=dw1.permute(2, 1, 0),                     # (E, H, I), M-major
        dw2_out=dw2.permute(2, 0, 1),                     # (E, H, I), N-major
    )


class TuningSpecTest(TestCommons):
    def test_spec_parse_roundtrip(self):
        for text, expect in [
            ("128x320 coop c2x1", ConfigSpec(128, 320, None, False, 2, 1)),
            ("256x160x128 coop c1x1", ConfigSpec(256, 160, 128, False, 1, 1)),
            ("128x160 pingpong c1x2 swz4", ConfigSpec(128, 160, None, True, 1, 2, 4)),
            ("64x128 pp", ConfigSpec(64, 128, None, True, 1, 1)),
            ("128x192 swap", ConfigSpec(128, 192, None, False, 1, 1, 8, True)),
        ]:
            spec = ConfigSpec.parse(text)
            self.assertEqual(spec, expect, text)
            self.assertEqual(ConfigSpec.parse(str(spec)), spec, text)
        for bad in ["", "128", "128x320 c2", "128x320 fast"]:
            with self.assertRaises(ValueError):
                ConfigSpec.parse(bad)

    def test_spec_matches_quack_space(self):
        import quack.gemm_config as gc

        # every default SM90 config round-trips through describe/parse and lands back on the same object
        for c in tuning._default_space.get(9) or set(gc._get_sm90_configs()):
            self.assertEqual(ConfigSpec.parse(tuning.describe(c)).to_gemm_config(9), c)

    def test_expand_roles_merge(self):
        ov = tuning._expand_roles(
            {"all": {"dynamic_scheduler": True}, "dw": {"configs": ["128x320 coop c2x1"], "lpt_permute": True},
             "dact": {"dynamic_scheduler": False}}
        )
        self.assertEqual(set(ov), set(tuning.ROLES))
        self.assertEqual(ov["dw1"], RoleOverride((ConfigSpec(128, 320, None, False, 2, 1),), True, True))
        self.assertEqual(ov["dact"], RoleOverride(None, False, None))
        self.assertEqual(ov["down"], RoleOverride(None, True, None))
        self.assertEqual(tuning._expand_roles({}), {})
        with self.assertRaises(ValueError):
            tuning._expand_roles({"dw3": {}})
        with self.assertRaises(ValueError):
            tuning._expand_roles({"dw1": {"tile": 1}})

    def test_infer_role_gemm(self):
        v = _views()
        A = torch.empty(8, 16)
        cu = torch.zeros(17, dtype=torch.int32)
        self.assertEqual(tuning.infer_role_gemm({"A": A, "B": torch.empty(8, 16), "out": v["dw1_out"], "cu_seqlens_k": cu}), "dw1")
        self.assertEqual(tuning.infer_role_gemm({"A": A, "B": torch.empty(8, 16), "out": v["dw2_out"], "cu_seqlens_k": cu}), "dw2")
        self.assertEqual(tuning.infer_role_gemm({"A": A, "B": v["down_B"], "out": A, "cu_seqlens_m": cu}), "down")
        self.assertEqual(tuning.infer_role_gemm({"A": A, "B": v["dx_B"], "out": A, "cu_seqlens_m": cu}), "dx")
        self.assertIsNone(tuning.infer_role_gemm({"A": A, "B": v["dx_B"], "out": A, "cu_seqlens_m": cu, "A_idx": cu}))
        self.assertIsNone(tuning.infer_role_gemm({"A": A, "B": v["dx_B"], "out": A}))

    def test_lpt_permute(self):
        off = torch.tensor([0, 5, 5, 12, 20, 21, 30, 31, 40], dtype=torch.int32)
        tuning.set_overrides({"dw": {"lpt_permute": True}})
        try:
            perm = tuning.lpt_permute("dw1", off)
            self.assertEqual(perm.dtype, torch.int32)
            counts = (off[1:] - off[:-1])[perm.long()]
            self.assertTrue(torch.all(counts[:-1] >= counts[1:]))
            self.assertEqual(sorted(perm.tolist()), list(range(8)))
            self.assertIsNone(tuning.lpt_permute("dx", off))
            self.assertIsNone(tuning.lpt_permute("dw1", off[:7]), "E=6 is not a multiple of 4")
        finally:
            tuning.set_overrides(None)
        self.assertIsNone(tuning.lpt_permute("dw1", off))


class Sm90ShapeTest(TestCommons):
    def test_default_space_passes(self):
        import quack.gemm_config as gc

        for c in tuning._default_space.get(9) or set(gc._get_sm90_configs()):
            self.assertIsNone(tuning.sm90_spec_problem(ConfigSpec.from_gemm_config(c)), tuning.describe(c))

    def test_known_good_and_bad_shapes(self):
        ok = ["256x160 coop c1x1", "128x160 coop c2x1", "128x160x128 coop c2x1", "192x128 coop c2x1",
              "192x128 pingpong c2x1", "128x160x128 pingpong c2x1", "128x208 pingpong c2x1", "64x256 pingpong c1x1"]
        for s in ok:
            self.assertIsNone(tuning.sm90_spec_problem(ConfigSpec.parse(s)), s)
        bad = {"128x320 coop c2x1": "wgmma N", "256x160x128 coop c2x1": "pipeline stages", "256x320 coop c2x1": "wgmma N",
               "320x160 coop c2x1": "two warpgroups", "192x160 coop c2x1": "two warpgroups", "192x256 coop c2x1": "two warpgroups",
               "128x224 pingpong c2x1": "<= 208", "256x192 pingpong c2x1": "pingpong needs tile_m", "320x192 coop c2x1": "<= 160",
               "128x160x40 coop c2x1": "multiple of 16", "256x160 coop c4x2": "at most 4 CTAs"}
        for s, why in bad.items():
            problem = tuning.sm90_spec_problem(ConfigSpec.parse(s))
            self.assertIsNotNone(problem, s)
            self.assertIn(why, problem, s)

    def test_invalid_override_is_skipped_with_a_warning(self):
        import warnings as w
        from quack.autotuner import AutotuneConfig
        import quack.gemm_config as gc

        tuning.set_overrides({"dw1": {"configs": ["128x320 coop c2x1", "256x160 coop c1x1"]}})
        try:
            space = [AutotuneConfig(config=c) for c in gc._get_sm90_configs()]
            self.assertNotIn(ConfigSpec(128, 320, None, False, 2, 1).to_gemm_config(9), {c.kwargs["config"] for c in space})
            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                kept = tuning._filter_for_role(space, "dw1", 9, "test")
            self.assertEqual([tuning.describe(c.kwargs["config"]) for c in kept], ["256x160 coop c1x1"])
            self.assertTrue(any("wgmma N" in str(x.message) for x in caught))
        finally:
            tuning.set_overrides(None)


class OverrideTextTest(TestCommons):
    def test_three_forms(self):
        import tempfile

        self.assertIsNone(tuning.load_overrides_text(""))
        self.assertEqual(tuning.load_overrides_text('{"dw": {"lpt_permute": true}}'), {"dw": {"lpt_permute": True}})
        with tempfile.TemporaryDirectory() as d:
            plain = os.path.join(d, "ov.json")
            with open(plain, "w") as f:
                json.dump({"dx": {"dynamic_scheduler": True}}, f)
            self.assertEqual(tuning.load_overrides_text(plain), {"dx": {"dynamic_scheduler": True}})
            sweep = os.path.join(d, "sweep.json")
            with open(sweep, "w") as f:
                json.dump({"settings": [{"name": "default", "overrides": None},
                                        {"name": "best", "overrides": {"all": {"dynamic_scheduler": True}}}]}, f)
            self.assertEqual(tuning.load_overrides_text(sweep + ":best"), {"all": {"dynamic_scheduler": True}})
            self.assertIsNone(tuning.load_overrides_text(sweep + ":default"))
            with self.assertRaises(ValueError):
                tuning.load_overrides_text(sweep + ":missing")
            with self.assertRaises(ValueError):
                tuning.load_overrides_text(sweep)


class TuningSpaceTest(TestCommons):
    """The autotune-space filter, exercised on the SM90 lists without a GPU."""

    def setUp(self):
        import quack.gemm_config as gc
        from quack.autotuner import AutotuneConfig

        self.space = [AutotuneConfig(config=c) for c in gc._get_sm90_configs()]
        self.default = tuning._default_space[9]
        self.extras = tuning._extra_configs[9]
        self.assertGreater(len(self.extras), 0)
        self.assertTrue(self.default.isdisjoint(self.extras))

    def tearDown(self):
        tuning.set_overrides(None)

    def test_no_override_restores_default_space(self):
        for role in tuning.ROLES + (None,):
            kept = tuning._filter_for_role(self.space, role, 9, "test")
            self.assertEqual({c.kwargs["config"] for c in kept}, set(self.default), role)

    def test_override_restricts_to_listed_configs(self):
        tuning.set_overrides({"dw1": {"configs": ["128x160 coop c2x1", "256x160 coop c2x1"]}})
        kept = tuning._filter_for_role(self.space, "dw1", 9, "test")
        self.assertEqual({tuning.describe(c.kwargs["config"]) for c in kept}, {"128x160 coop c2x1", "256x160 coop c2x1"})
        kept = tuning._filter_for_role(self.space, "dw2", 9, "test")
        self.assertEqual({c.kwargs["config"] for c in kept}, set(self.default))

    def test_override_adds_unlisted_shape_to_space(self):
        import quack.gemm_config as gc

        tuning.set_overrides({"dact": {"configs": ["64x128 pingpong c1x1"]}})
        space = gc._get_sm90_configs()
        self.assertIn(ConfigSpec(64, 128, None, True, 1, 1).to_gemm_config(9), space)
        # and gemm_tuned's captured list follows
        import quack.gemm_interface as gi

        self.assertIn(ConfigSpec(64, 128, None, True, 1, 1).to_gemm_config(9), {c.kwargs["config"] for c in gi.gemm_tuned.configs})

    def test_override_pruned_away_raises(self):
        tuning.set_overrides({"dw1": {"configs": ["256x160 coop c1x2"]}})   # gather_A forbids cluster_n > 1
        from quack.autotuner import AutotuneConfig

        kept_by_quack = [c for c in self.space if c.kwargs["config"].cluster_n == 1]
        with self.assertRaises(ValueError):
            tuning._filter_for_role(kept_by_quack, "dw1", 9, "test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TuningGpuTest(TestCommons):
    @classmethod
    def setUpClass(cls):
        cls.default = run_child({"SONICMOE_TUNING": "", "SONICMOE_PROFILE_RANGES": "1"}, want_profile=True)

    def test_default_bit_identical_to_pristine(self):
        pristine = run_child({"SONICMOE_TUNING": "", "SONICMOE_TUNING_HOOKS": "0"})
        self.assertFalse(pristine["hooks"])
        self.assertTrue(self.default["hooks"])
        for name in ("y", "dx", "dw1", "dw2"):
            self.assertTrue(torch.equal(self.default[name], pristine[name]), name)

    def test_winners_cover_every_role(self):
        self.assertEqual(set(self.default["winners"]), set(tuning.ROLES))
        for role, w in self.default["winners"].items():
            ConfigSpec.parse(w["config"])        # a well-formed spec
            self.assertIn("timings_ms", w, role)  # a fresh process tunes (or reads the disk cache): the table is there

    def test_ranges_attribute_every_role(self):
        rows = self.default["range_ms"]
        for name in tuning.RANGE_NAMES:
            self.assertIn(f"sonicmoe/{name}", rows, name)
            self.assertGreater(rows[f"sonicmoe/{name}"], 0.0, name)

    def test_lpt_permute_bit_exact(self):
        res = run_child({"SONICMOE_TUNING": json.dumps({"dw": {"lpt_permute": True}})})
        for name in ("y", "dx", "dw1", "dw2"):
            self.assertTrue(torch.equal(self.default[name], res[name]), name)

    def test_single_config_override_runs(self):
        # pin dw1 to its current winner: same kernel, bit-exact, and the space collapses to one entry
        winner = self.default["winners"]["dw1"]["config"]
        res = run_child({"SONICMOE_TUNING": json.dumps({"dw1": {"configs": [winner]}})})
        self.assertEqual(res["winners"]["dw1"]["config"], winner)
        self.assertEqual(list(res["winners"]["dw1"]["timings_ms"]), [winner])
        for name in ("y", "dx", "dw1", "dw2"):
            self.assertTrue(torch.equal(self.default[name], res[name]), name)
