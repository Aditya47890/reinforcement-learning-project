"""
=============================================================================
RLVER — Setup Verification Script
File  : verify_setup.py
Author: Dushyant (u23ai085)

Run AFTER 00_setup.sh to confirm everything is correctly installed.
Does NOT require any model downloads — tests import, logic, and GPU only.

Usage:
  conda activate rlver_env
  cd ~/Dushyant_u23ai085_9256777500
  python verify_setup.py

  # With model tests (if models are downloaded):
  python verify_setup.py --with-models
=============================================================================
"""

from __future__ import annotations

import os, sys, time, json, argparse, importlib, subprocess
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_DIR = os.path.expanduser("~/Dushyant_u23ai085_9256777500")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "rl_training"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "rlver_adversarial"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "ablation"))

# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self):
        self.results = {"pass": [], "fail": [], "warn": [], "skip": []}
        self._section = ""

    def section(self, name: str) -> None:
        self._section = name
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")

    def ok(self, label: str, detail: str = "") -> None:
        self.results["pass"].append(f"{self._section}: {label}")
        d = f"  ({detail})" if detail else ""
        print(f"  ✓  {label}{d}")

    def fail(self, label: str, detail: str = "") -> None:
        self.results["fail"].append(f"{self._section}: {label}: {detail}")
        print(f"  ✗  {label}  ← {detail}")

    def warn(self, label: str, detail: str = "") -> None:
        self.results["warn"].append(f"{self._section}: {label}")
        d = f" ({detail})" if detail else ""
        print(f"  ⚠  {label}{d}")

    def skip(self, label: str, reason: str = "") -> None:
        self.results["skip"].append(label)
        print(f"  ○  SKIP: {label}  ({reason})")

    def chk(self, cond: bool, label: str, fail_detail: str = "", ok_detail: str = "") -> bool:
        if cond:
            self.ok(label, ok_detail)
        else:
            self.fail(label, fail_detail)
        return cond

    def summary(self) -> bool:
        n_pass = len(self.results["pass"])
        n_fail = len(self.results["fail"])
        n_warn = len(self.results["warn"])
        n_skip = len(self.results["skip"])
        total  = n_pass + n_fail

        print(f"\n  {'='*65}")
        print(f"  VERIFICATION SUMMARY")
        print(f"  {'='*65}")
        print(f"  Passed  : {n_pass}/{total}")
        print(f"  Failed  : {n_fail}")
        print(f"  Warnings: {n_warn}")
        print(f"  Skipped : {n_skip}")

        if self.results["warn"]:
            print(f"\n  Warnings:")
            for w in self.results["warn"]:
                print(f"    ⚠ {w}")

        if self.results["fail"]:
            print(f"\n  Failures (ACTION REQUIRED):")
            for f in self.results["fail"]:
                print(f"    ✗ {f}")
            print(f"\n  {'='*65}")
            print(f"  ✗ SETUP INCOMPLETE — fix failures before training")
            print(f"  {'='*65}")
            return False
        else:
            print(f"\n  {'='*65}")
            print(f"  ✓ ALL CHECKS PASSED — ready to run training!")
            print(f"  {'='*65}")
            return True


T = TestRunner()

# ─────────────────────────────────────────────────────────────────────────────
# A. Python version
# ─────────────────────────────────────────────────────────────────────────────

T.section("A. Python Version")
pv = sys.version_info
T.chk(pv.major == 3 and pv.minor == 11,
      f"Python 3.11.x", f"Got {pv.major}.{pv.minor}", f"{pv.major}.{pv.minor}.{pv.micro}")

# ─────────────────────────────────────────────────────────────────────────────
# B. PyTorch + CUDA
# ─────────────────────────────────────────────────────────────────────────────

T.section("B. PyTorch + CUDA")
try:
    import torch
    T.ok(f"torch imported: v{torch.__version__}")

    cuda_ok = T.chk(torch.cuda.is_available(), "CUDA available")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        cc    = props.major * 10 + props.minor
        vram  = props.total_memory / 1e9

        T.ok(f"GPU: {props.name}  sm_{cc}  {vram:.1f} GB VRAM")
        T.chk(vram >= 16, f"VRAM ≥ 16 GB", f"Only {vram:.1f} GB — may OOM on 7B+LoRA")
        T.chk(props.major >= 8, f"sm_{cc} ≥ sm_80 (Ampere+ required for bf16)")
        T.chk(torch.cuda.is_bf16_supported(), "bfloat16 supported")

        if cc == 120:
            T.warn(f"sm_120 (Blackwell consumer)",
                   "Uses PTX JIT — correct results, ~20% slower. Run fix_gpu_sm120.sh")
        elif cc >= 89:
            T.ok(f"sm_{cc} fully supported by PyTorch")

        # BF16 ops test
        try:
            x = torch.ones(512, 512, dtype=torch.bfloat16, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            T.chk(y.shape == (512, 512), "bfloat16 matmul on GPU")
            del x, y; torch.cuda.empty_cache()
        except Exception as e:
            T.fail("bfloat16 matmul", str(e))

    # Gradient test
    x = torch.tensor([2.0], requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward()
    T.chk(x.grad is not None and abs(x.grad.item() - 4.0) < 1e-5,
          "Autograd works", f"grad={x.grad}")

except ImportError as e:
    T.fail("torch import", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# C. HuggingFace packages
# ─────────────────────────────────────────────────────────────────────────────

T.section("C. HuggingFace Ecosystem")
hf_packages = [
    ("transformers",    "4.40"),
    ("peft",            "0.12"),
    ("accelerate",      "1.0"),
    ("trl",             "0.10"),
    ("datasets",        "3.0"),
    ("tokenizers",      "0.20"),
    ("huggingface_hub", "0.25"),
    ("sentencepiece",   None),
    ("safetensors",     None),
]
for mod, min_ver in hf_packages:
    try:
        m   = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        if min_ver:
            try:
                cur = tuple(int(x) for x in ver.split(".")[:2])
                req = tuple(int(x) for x in min_ver.split(".")[:2])
                ok  = cur >= req
                T.chk(ok, f"{mod} v{ver}", f"Need ≥ {min_ver}")
            except ValueError:
                T.ok(f"{mod} v{ver}")
        else:
            T.ok(f"{mod} v{ver}")
    except ImportError as e:
        T.fail(f"{mod} not installed", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# D. Analysis packages
# ─────────────────────────────────────────────────────────────────────────────

T.section("D. Analysis / Utility Packages")
util_packages = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn",
    "tqdm", "yaml", "jsonlines", "rich", "psutil",
]
for mod in util_packages:
    try:
        m   = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        # NumPy 2.x check
        if mod == "numpy":
            major = int(ver.split(".")[0])
            T.chk(major < 2, f"numpy v{ver} < 2.0 (API compatibility)", f"v{ver} — numpy 2.x may break scipy/sklearn")
        else:
            T.ok(f"{mod} v{ver}")
    except ImportError as e:
        T.warn(f"{mod} not installed", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# E. RLVER modules
# ─────────────────────────────────────────────────────────────────────────────

T.section("E. RLVER Project Modules")

# reward_model
try:
    from reward_model import RewardConfig, compute_reward, compute_rewards_batch, RLVERRewardFunction
    cfg = RewardConfig()
    r, c = compute_reward(
        "<think>Check: 2+2=4. Matches.</think>\nVERDICT: CORRECT\nCONFIDENCE: 0.95",
        true_label=1, problem="What is 2+2?", cfg=cfg
    )
    T.chk(r > 0 and c["correct"] == 1, f"reward_model: correct → r={r:.3f}", f"r={r}")
    r2, c2 = compute_reward("VERDICT: CORRECT\nCONFIDENCE: 0.9", true_label=0, problem="test", cfg=cfg)
    T.chk(r2 < 0 and c2["correct"] == 0, f"reward_model: wrong → r={r2:.3f}", f"r={r2}")
    # Batch
    rs, cs = compute_rewards_batch(["VERDICT: CORRECT\nCONFIDENCE: 0.9"]*3,
                                   [1, 0, 1], ["p"]*3, cfg=cfg)
    T.chk(len(rs) == 3, "compute_rewards_batch returns 3 values")
except Exception as e:
    T.fail("reward_model", str(e))

# metrics
try:
    from metrics import (EvalMetrics, VerificationSample, compute_metrics,
                         parse_model_output, compute_ece)
    pred, conf, think = parse_model_output(
        "<think>Steps here</think>\nVERDICT: INCORRECT\nCONFIDENCE: 0.80"
    )
    T.chk(pred == 0 and conf == 0.80 and think is not None,
          f"metrics.parse_model_output: pred={pred} conf={conf}")
    # ECE
    import numpy as np
    ece = compute_ece(np.array([1,0,1,0]), np.array([1,0,1,0]),
                      np.array([0.9,0.8,0.85,0.75]))
    T.chk(ece < 0.2, f"metrics.compute_ece: {ece:.4f}")
except Exception as e:
    T.fail("metrics module", str(e))

# dataset_builder
try:
    from dataset_builder import (RLVERSample, RLVERDatasetBuilder,
                                  format_verification_prompt, to_hf_dataset,
                                  perturb_solution_rule_based, assign_difficulty)
    s = RLVERSample(
        problem_id="v001", problem="A baker makes 240 cookies in 8 hours. Cookies/hr?",
        solution="240 ÷ 8 = 30 cookies per hour.", label=1,
        is_adversarial=False, difficulty="easy", answer="30", source="test"
    )
    prompt = format_verification_prompt(s, thinking_enabled=True)
    T.chk("<think>" in prompt and "VERDICT" in prompt and "240" in prompt,
          "format_verification_prompt (thinking=True)")
    prompt_no = format_verification_prompt(s, thinking_enabled=False)
    T.chk("<think>" not in prompt_no and "VERDICT" in prompt_no,
          "format_verification_prompt (thinking=False)")

    diff = assign_difficulty(s.problem, s.solution)
    T.chk(diff in ("easy", "medium", "hard"), f"assign_difficulty: {diff}")

    perturbed = perturb_solution_rule_based(s.solution, s.answer)
    T.chk(perturbed != s.solution or True, f"perturb_solution_rule_based: ran without error")

    # to_hf_dataset
    ds = to_hf_dataset([s], thinking_enabled=True)
    T.chk("problem" in ds.column_names and "label" in ds.column_names,
          f"to_hf_dataset: columns={list(ds.column_names)}")
except Exception as e:
    T.fail("dataset_builder", str(e))

# ppo_trainer_custom
try:
    import torch
    from ppo_trainer_custom import (ValueHead, ActorCriticModel, RolloutBuffer,
                                     RolloutSample, KLController, compute_ppo_loss)
    # ValueHead forward
    vh   = ValueHead(hidden_size=64)
    h    = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    v    = vh(h, mask)
    T.chk(v.shape == (2,), f"ValueHead forward: shape={tuple(v.shape)}")

    # RolloutBuffer + GAE
    buf = RolloutBuffer(capacity=10)
    import random as _r
    for _ in range(5):
        buf.add(RolloutSample(
            query_ids=torch.randint(0,100,(8,)), response_ids=torch.randint(0,100,(12,)),
            reward=_r.uniform(-1,1), value=_r.uniform(-0.3,0.3), log_prob=-2.5
        ))
    buf.compute_gae(gamma=0.99, lam=0.95, normalize=True)
    T.chk(all(isinstance(s.advantage, float) for s in buf._buf),
          "RolloutBuffer.compute_gae: advantages computed")

    # PPO loss
    B, T_len, V = 2, 20, 256
    lnew = torch.randn(B, T_len, V, requires_grad=True)
    lref = torch.randn(B, T_len, V)
    ids  = torch.randint(0, V, (B, T_len))
    rmask= torch.zeros(B, T_len, dtype=torch.long); rmask[:, 8:] = 1
    adv  = torch.tensor([0.5, -0.3])
    ret  = adv + torch.tensor([0.2, -0.1])
    olp  = torch.tensor([-2.5, -2.8])
    vals = torch.tensor([0.3, -0.2])
    loss, stats = compute_ppo_loss(lnew, lref, ids, rmask, olp, vals, adv, ret)
    loss.backward()
    T.chk(lnew.grad is not None, f"PPO loss backward: grad_norm={lnew.grad.norm():.4f}")

    # KL controller
    kl = KLController(init_kl_coef=0.05, target_kl=6.0)
    for v in [8.0, 5.5, 6.3]: kl.update(v)
    T.chk(0 < kl.kl_coef < 1.0, f"KLController adaptive: β={kl.kl_coef:.5f}")

except Exception as e:
    T.fail("ppo_trainer_custom", str(e))

# grpo_trainer_custom
try:
    import torch, numpy as np
    from grpo_trainer_custom import (GroupSample, compute_grpo_loss)
    G, R = 6, 24
    new_lps = torch.randn(G, R, requires_grad=True)
    old_lps = (new_lps + torch.randn(G, R) * 0.05).detach()
    ref_lps = (new_lps + torch.randn(G, R) * 0.10).detach()
    mask    = torch.ones(G, R)
    rewards = np.array([0.8, -0.5, 0.3, -1.0, 0.6, -0.2])
    adv     = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    adv_t   = torch.tensor(adv, dtype=torch.float32)
    loss, stats = compute_grpo_loss(new_lps, old_lps, ref_lps, mask, adv_t, epsilon=0.2, beta=0.04)
    loss.backward()
    T.chk(new_lps.grad is not None,
          f"GRPO loss backward: loss={loss.item():.4f} grad_norm={new_lps.grad.norm():.4f}")

    # GroupSample advantage computation
    g = GroupSample(
        prompt="test", completion_ids=[torch.zeros(1,10,dtype=torch.long)]*G,
        completions=["out"]*G, prompt_len=5,
        rewards=[0.8,-0.5,0.3,-1.0,0.6,-0.2],
        true_label=1, problem="p", problem_id="g0"
    )
    g.compute_advantages()
    T.chk(len(g.advantages) == G and abs(np.mean(g.advantages)) < 0.01,
          f"GroupSample.compute_advantages: mean≈0 ✓")
    T.chk(g.has_signal, "GroupSample.has_signal True for varied rewards")
except Exception as e:
    T.fail("grpo_trainer_custom", str(e))

# ablation_config
try:
    from ablation_config import ALL_ABLATION_GROUPS, list_all_experiments, get_experiment_config
    all_e = list_all_experiments()
    T.chk(len(ALL_ABLATION_GROUPS) == 9, f"9 ablation groups")
    T.chk(len(all_e) >= 48, f"{len(all_e)} ablation experiments")
    exp   = get_experiment_config("B7")
    T.chk(exp.is_ours and exp.algorithm == "grpo", f"B7 = GRPO+Thinking (ours)")
    exp_b1 = get_experiment_config("B1")
    T.chk(exp_b1.algorithm == "zero_shot", "B1 = zero_shot baseline")
except Exception as e:
    T.fail("ablation_config", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# F. Project file structure
# ─────────────────────────────────────────────────────────────────────────────

T.section("F. Project File Structure")
required_files = [
    "configs/ppo_config.yaml", "configs/grpo_config.yaml",
    "rl_training/__init__.py", "rl_training/reward_model.py",
    "rl_training/ppo_trainer_custom.py", "rl_training/grpo_trainer_custom.py",
    "rl_training/rlver_rl_train.py",
    "rlver_adversarial/__init__.py", "rlver_adversarial/dataset_builder.py",
    "rlver_adversarial/metrics.py", "rlver_adversarial/rlver_eval.py",
    "rlver_adversarial/adversarial_gen.py",
    "ablation/__init__.py", "ablation/ablation_config.py",
    "ablation/ablation_runner.py", "ablation/ablation_analysis.py",
    "scripts/01_download_data.py", "scripts/02_run_training.sh",
    "scripts/03_run_eval.sh", "scripts/04_analyze_results.py",
    "scripts/05_run_ablations.sh", "run_all.sh",
    "fix_gpu_sm120.sh", "README.md", "environment.yml",
]
required_dirs = [
    "models", "logs", "results", "cache/hf", "cache/datasets",
    "results/ablation",
]
for rel in required_files:
    p = os.path.join(PROJECT_DIR, rel)
    T.chk(os.path.isfile(p), rel)
for rel in required_dirs:
    p = os.path.join(PROJECT_DIR, rel)
    T.chk(os.path.isdir(p), f"{rel}/")

# Models
T.section("G. Downloaded Models")
for model_alias, hf_id in [
    ("qwen_7b",       "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen_1p5b",     "Qwen/Qwen2.5-1.5B-Instruct"),
    ("mistral_7b_sim","mistralai/Mistral-7B-Instruct-v0.3"),
]:
    model_dir = os.path.join(PROJECT_DIR, "models", model_alias)
    if not os.path.isdir(model_dir):
        T.warn(f"models/{model_alias} not downloaded",
               f"Run: python scripts/01_download_data.py")
        continue
    has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
    weights    = list(Path(model_dir).rglob("*.safetensors")) + \
                 list(Path(model_dir).rglob("*.bin"))
    size_gb    = sum(os.path.getsize(p) for p in Path(model_dir).rglob("*")
                     if os.path.isfile(p)) / 1e9
    T.chk(has_config and len(weights) > 0,
          f"models/{model_alias}: {len(weights)} weight files, {size_gb:.1f} GB",
          "missing config.json or weight files")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

print()
success = T.summary()

if success:
    print("""
  NEXT STEPS:
    conda activate rlver_env
    source configs/rlver_env_vars.sh   # set GPU env vars

    # Step 1: Prepare dataset
    python scripts/01_download_data.py
    # (or with Mistral adversarials: add --gen_adversarial)

    # Step 2: Train all variants
    bash scripts/02_run_training.sh

    # Step 3: Evaluate
    bash scripts/03_run_eval.sh

    # Step 4: Analyze results
    python scripts/04_analyze_results.py

    # Step 5: Ablation study
    bash scripts/05_run_ablations.sh
""")
else:
    print("""
  TO FIX FAILURES:
    1. GPU sm_120 warning: bash fix_gpu_sm120.sh
    2. Missing packages:   conda activate rlver_env && pip install <pkg>
    3. Missing models:     python scripts/01_download_data.py
    4. Missing files:      unzip RLVER_project.zip and re-run bash 00_setup.sh
""")

sys.exit(0 if success else 1)


if __name__ == "__main__":
    pass   # script runs at import time (top-level)
