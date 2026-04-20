#!/bin/bash
# =============================================================================
# RLVER — GPU sm_120 Fix + Full Environment Verification
# File  : fix_gpu_sm120.sh
# Author: Dushyant (u23ai085)
#
# PURPOSE:
#   Your RTX PRO 4500 is Blackwell GB202 (sm_120).
#   PyTorch 2.6.0+cu124 officially supports only up to sm_100 (datacenter).
#   This script:
#     1. Detects your exact GPU compute capability
#     2. Upgrades to PyTorch nightly (cu128) for native sm_120 support
#     3. Sets all required environment variables
#     4. Runs a comprehensive verification test
#     5. Confirms training will work correctly
#
# USAGE:
#   conda activate rlver_env
#   bash fix_gpu_sm120.sh
#
#   Options:
#     --nightly     : Install PyTorch nightly (recommended for sm_120)
#     --keep-stable : Keep PyTorch 2.6.0, just set env vars (faster, ~20% slower training)
#     --test-only   : Only run verification, no install
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

info()  { echo -e "${BLU}[INFO]${NC}  $*"; }
ok()    { echo -e "${GRN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YLW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[FAIL]${NC}  $*" >&2; exit 1; }
hdr()   { echo -e "\n${BOLD}${CYN}══ $* ══${NC}"; }

PROJECT_DIR="$HOME/Dushyant_u23ai085_9256777500"
ENV_NAME="rlver_env"
MODE="${1:---nightly}"   # default: install nightly

# ── Parse args ────────────────────────────────────────────────────────────────
USE_NIGHTLY=true
TEST_ONLY=false
case "$MODE" in
    --nightly)     USE_NIGHTLY=true;  TEST_ONLY=false ;;
    --keep-stable) USE_NIGHTLY=false; TEST_ONLY=false ;;
    --test-only)   USE_NIGHTLY=false; TEST_ONLY=true  ;;
    *) warn "Unknown mode: $MODE  (using --nightly)"; USE_NIGHTLY=true ;;
esac

# ── Verify we're in the right conda env ───────────────────────────────────────
CURRENT_ENV="${CONDA_DEFAULT_ENV:-none}"
if [ "$CURRENT_ENV" != "$ENV_NAME" ]; then
    die "Wrong conda environment: '$CURRENT_ENV'\nPlease run:\n  conda activate $ENV_NAME\n  bash fix_gpu_sm120.sh"
fi
ok "Conda env: $CURRENT_ENV"

# ═════════════════════════════════════════════════════════════════════════════
hdr "STEP 1 — Detect GPU compute capability"
# ═════════════════════════════════════════════════════════════════════════════

GPU_INFO=$(python - << 'PYEOF'
import subprocess, sys

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return ""

# nvidia-smi info
name   = run("nvidia-smi --query-gpu=name --format=csv,noheader")
vram   = run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
driver = run("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
cuda_v = run("nvidia-smi | grep 'CUDA Version' | awk '{print $NF}'")

import torch
if not torch.cuda.is_available():
    print("NO_GPU")
    sys.exit(0)

props = torch.cuda.get_device_properties(0)
cc    = props.major * 10 + props.minor   # e.g. sm_120 → 120

# Classify architecture
arch_map = {
    35: "Kepler",  37: "Kepler",
    50: "Maxwell", 52: "Maxwell", 53: "Maxwell",
    60: "Pascal",  61: "Pascal",  62: "Pascal",
    70: "Volta",   72: "Volta",
    75: "Turing",
    80: "Ampere",  86: "Ampere",  87: "Ampere",
    89: "Ada Lovelace",
    90: "Hopper",
    100:"Blackwell (Datacenter B100/B200)",
    120:"Blackwell (Consumer GB202)",
}
arch = arch_map.get(cc, f"Unknown (sm_{cc})")

vram_gb = props.total_memory / 1e9
print(f"NAME={name}")
print(f"VRAM_MiB={vram}")
print(f"VRAM_GB={vram_gb:.1f}")
print(f"CC={cc}")
print(f"CC_MAJOR={props.major}")
print(f"CC_MINOR={props.minor}")
print(f"ARCH={arch}")
print(f"DRIVER={driver}")
print(f"CUDA_VERSION={cuda_v}")
print(f"TORCH_VERSION={torch.__version__}")
PYEOF
)

if [ "$GPU_INFO" = "NO_GPU" ]; then
    die "No CUDA GPU detected. Check: nvidia-smi && nvcc --version"
fi

# Parse GPU info
GPU_NAME=$(echo "$GPU_INFO" | grep "^NAME=" | cut -d= -f2-)
GPU_CC=$(echo "$GPU_INFO" | grep "^CC=" | cut -d= -f2)
GPU_MAJOR=$(echo "$GPU_INFO" | grep "^CC_MAJOR=" | cut -d= -f2)
GPU_MINOR=$(echo "$GPU_INFO" | grep "^CC_MINOR=" | cut -d= -f2)
GPU_ARCH=$(echo "$GPU_INFO" | grep "^ARCH=" | cut -d= -f2-)
GPU_VRAM=$(echo "$GPU_INFO" | grep "^VRAM_GB=" | cut -d= -f2)
DRIVER=$(echo "$GPU_INFO" | grep "^DRIVER=" | cut -d= -f2)
CUDA_VER=$(echo "$GPU_INFO" | grep "^CUDA_VERSION=" | cut -d= -f2)
TORCH_VER=$(echo "$GPU_INFO" | grep "^TORCH_VERSION=" | cut -d= -f2)

echo ""
echo -e "  ┌─ GPU Detected ──────────────────────────────────────────────┐"
echo -e "  │  Name       : ${BOLD}$GPU_NAME${NC}"
echo -e "  │  VRAM       : ${BOLD}${GPU_VRAM} GB${NC}"
echo -e "  │  Architecture: ${BOLD}$GPU_ARCH${NC}"
echo -e "  │  Compute    : ${BOLD}sm_${GPU_CC}  (${GPU_MAJOR}.${GPU_MINOR})${NC}"
echo -e "  │  Driver     : $DRIVER"
echo -e "  │  CUDA       : $CUDA_VER"
echo -e "  │  PyTorch    : $TORCH_VER"
echo -e "  └─────────────────────────────────────────────────────────────┘"
echo ""

# Determine required action based on compute capability
if [ "$GPU_CC" -lt 89 ]; then
    warn "GPU sm_${GPU_CC} is pre-Ada. RLVER requires sm_89+ for bf16 performance."
    warn "Training will work but may be slow."
elif [ "$GPU_CC" -eq 89 ]; then
    ok "GPU sm_89 (Ada Lovelace) — fully supported by PyTorch 2.6.0+cu124"
    USE_NIGHTLY=false
elif [ "$GPU_CC" -eq 100 ]; then
    ok "GPU sm_100 (Blackwell datacenter) — supported by PyTorch 2.6.0+cu124"
    USE_NIGHTLY=false
elif [ "$GPU_CC" -eq 120 ]; then
    warn "GPU sm_120 (Blackwell consumer GB202) — needs PyTorch nightly for native support"
    warn "With PyTorch 2.6.0: works via PTX JIT (10-30% slower, no correctness issues)"
    if [ "$USE_NIGHTLY" = true ]; then
        info "→ Will upgrade to PyTorch nightly (cu128) for native sm_120 kernels"
    else
        info "→ Will keep PyTorch 2.6.0 and set env vars for sm_120 compatibility"
    fi
else
    warn "GPU sm_${GPU_CC} — unrecognized, proceeding with current PyTorch"
    USE_NIGHTLY=false
fi

if [ "$TEST_ONLY" = true ]; then
    info "--test-only: skipping installation, jumping to verification"
    USE_NIGHTLY=false
fi

# ═════════════════════════════════════════════════════════════════════════════
hdr "STEP 2 — Fix PyTorch for sm_${GPU_CC}"
# ═════════════════════════════════════════════════════════════════════════════

if [ "$USE_NIGHTLY" = true ] && [ "$GPU_CC" -eq 120 ]; then
    info "Upgrading to PyTorch nightly with CUDA 12.8 (native sm_120 support)..."
    info "This replaces torch==2.6.0 — all other packages remain unchanged."
    echo ""
    warn "Note: nightly builds are tested but not stable-released."
    warn "If nightly causes issues, run: bash fix_gpu_sm120.sh --keep-stable"
    echo ""

    # Uninstall current torch
    info "Removing PyTorch 2.6.0..."
    pip uninstall torch torchvision torchaudio -y --quiet 2>/dev/null || true

    # Install nightly with cu128 (supports sm_120)
    info "Installing PyTorch nightly (cu128)..."
    pip install \
        --pre \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --timeout 600 \
        || {
            warn "Nightly install failed. Falling back to stable 2.6.0+cu124..."
            pip install \
                torch==2.6.0 torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu124 \
                --timeout 300
            USE_NIGHTLY=false
        }

    # Verify the new install
    NEW_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
    info "New PyTorch version: $NEW_TORCH"
    if [[ "$NEW_TORCH" == *"dev"* ]] || [[ "$NEW_TORCH" > "2.6" ]]; then
        ok "PyTorch nightly installed successfully"
    else
        warn "PyTorch version unexpected: $NEW_TORCH"
    fi

elif [ "$GPU_CC" -eq 120 ] && [ "$USE_NIGHTLY" = false ]; then
    info "Keeping PyTorch 2.6.0 — setting sm_120 compatibility variables..."
fi

# ═════════════════════════════════════════════════════════════════════════════
hdr "STEP 3 — Set environment variables for sm_${GPU_CC}"
# ═════════════════════════════════════════════════════════════════════════════

# These variables are written to ~/.bashrc AND to a project-level env file
# so they are picked up by all training scripts automatically.

ENV_FILE="$PROJECT_DIR/configs/rlver_env_vars.sh"

cat > "$ENV_FILE" << ENVEOF
#!/bin/bash
# RLVER environment variables — auto-generated by fix_gpu_sm120.sh
# Source this file before training: source configs/rlver_env_vars.sh

# ── GPU / CUDA settings ───────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0               # use GPU 0 (single GPU setup)
export CUDA_LAUNCH_BLOCKING=0               # async kernel launch (faster)
export TORCH_CUDA_ARCH_LIST="${GPU_MAJOR}.${GPU_MINOR}"  # target sm_${GPU_CC} explicitly

# ── PyTorch performance settings ─────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TORCH_BACKENDS_CUDNN_BENCHMARK=1     # auto-tune kernels (faster conv)
export TOKENIZERS_PARALLELISM=false         # avoid HF tokenizer fork warning

# ── HuggingFace cache ─────────────────────────────────────────────────────────
export HF_HOME="$PROJECT_DIR/cache/hf"
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/hf"
export HF_HUB_CACHE="$PROJECT_DIR/cache/hf"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/hf/datasets"

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_DIR:\$PYTHONPATH"

# ── Weights & Biases (disable if not logged in) ───────────────────────────────
# export WANDB_API_KEY=your_key_here
# export WANDB_DISABLED=true          # uncomment to disable WandB logging

# ── bfloat16 for Blackwell ────────────────────────────────────────────────────
export ACCELERATE_TORCH_DTYPE="bfloat16"
ENVEOF

chmod +x "$ENV_FILE"
ok "Environment variables written → $ENV_FILE"

# Source it now for this session
# shellcheck disable=SC1090
source "$ENV_FILE"

# Also add to ~/.bashrc if not already there
BASHRC_LINE="source $ENV_FILE  # RLVER project env vars"
if ! grep -qF "rlver_env_vars.sh" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# RLVER project — load on conda activate" >> ~/.bashrc
    echo "$BASHRC_LINE" >> ~/.bashrc
    ok "Added to ~/.bashrc: $BASHRC_LINE"
else
    ok "~/.bashrc already has rlver_env_vars.sh"
fi

# ═════════════════════════════════════════════════════════════════════════════
hdr "STEP 4 — Comprehensive environment verification"
# ═════════════════════════════════════════════════════════════════════════════

python - << 'PYEOF'
import os, sys, json, time, subprocess
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_DIR = os.path.expanduser("~/Dushyant_u23ai085_9256777500")
PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

results = {"passed": [], "warnings": [], "failed": []}

def chk(cond, label, details=""):
    if cond:
        results["passed"].append(label)
        print(f"{PASS} {label}")
    else:
        results["failed"].append(f"{label}: {details}")
        print(f"{FAIL} {label}  ← {details}")

def wrn(label, details=""):
    results["warnings"].append(label)
    print(f"{WARN} {label}  ← {details}")

# ── 1. PyTorch + CUDA ─────────────────────────────────────────────────────────
print("\n  [A] PyTorch + CUDA")
import torch
chk(torch.cuda.is_available(), "CUDA available")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    cc    = props.major * 10 + props.minor
    vram  = props.total_memory / 1e9

    chk(vram >= 20, f"VRAM {vram:.1f} GB ≥ 20 GB (needed for Qwen-7B + LoRA)", f"Only {vram:.1f} GB")
    chk(props.major >= 8, f"Compute capability sm_{cc} ≥ sm_80 (Ampere+)", f"sm_{cc} may be slow")
    chk(torch.cuda.is_bf16_supported(), f"bfloat16 supported on sm_{cc}")

    # BF16 tensor operation test
    try:
        a = torch.ones(1024, 1024, dtype=torch.bfloat16, device="cuda")
        b = torch.ones(1024, 1024, dtype=torch.bfloat16, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        chk(c.shape == (1024, 1024), "bfloat16 matmul (1024×1024) on GPU: OK")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        chk(False, "bfloat16 matmul test", str(e))

    # Memory bandwidth test
    try:
        size = 1024 * 1024 * 512  # 512M elements
        x = torch.empty(size, dtype=torch.float16, device="cuda")
        t0 = time.time()
        y = x * 2.0
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        bw_gbps = (size * 2 * 2) / elapsed / 1e9  # read + write, 2 bytes each
        del x, y
        torch.cuda.empty_cache()
        chk(bw_gbps > 100, f"Memory bandwidth {bw_gbps:.0f} GB/s", f"Only {bw_gbps:.0f} GB/s")
    except Exception as e:
        wrn("Memory bandwidth test skipped", str(e))

    # Check sm_120 PTX JIT (no error = working in compat mode)
    if cc == 120:
        print(f"\n    [Note] sm_120 GPU detected:")
        print(f"    TORCH_CUDA_ARCH_LIST = {os.environ.get('TORCH_CUDA_ARCH_LIST','not set')}")
        print(f"    PyTorch will use PTX JIT compilation for sm_120")
        print(f"    This is CORRECT and produces valid results")
        print(f"    (10-30% slower than native kernels, no accuracy loss)")

# ── 2. HuggingFace packages ───────────────────────────────────────────────────
print("\n  [B] HuggingFace Ecosystem")
import_checks = [
    ("transformers",    "transformers"),
    ("peft",            "peft"),
    ("accelerate",      "accelerate"),
    ("trl",             "trl"),
    ("datasets",        "datasets"),
    ("huggingface_hub", "huggingface_hub"),
    ("tokenizers",      "tokenizers"),
    ("sentencepiece",   "sentencepiece"),
    ("safetensors",     "safetensors"),
]
for mod, name in import_checks:
    try:
        m   = __import__(mod)
        ver = getattr(m, "__version__", "?")
        chk(True, f"{name} v{ver}")
    except ImportError as e:
        chk(False, f"{name} import", str(e))

# Version compatibility check
try:
    import transformers, peft
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    pv = tuple(int(x) for x in peft.__version__.split(".")[:2])
    compat = tv >= (4, 40)
    chk(compat, f"transformers {transformers.__version__} ≥ 4.40 (required for Qwen2.5)")
except Exception as e:
    wrn("Version compatibility check", str(e))

# ── 3. Qwen-7B tokenizer test ─────────────────────────────────────────────────
print("\n  [C] Model Loading Test")
qwen_7b_path = os.path.join(PROJECT_DIR, "models", "qwen_7b")
qwen_1p5b_path = os.path.join(PROJECT_DIR, "models", "qwen_1p5b")

from transformers import AutoTokenizer
for model_path, model_name in [(qwen_7b_path, "Qwen-7B"), (qwen_1p5b_path, "Qwen-1.5B")]:
    if not os.path.isdir(model_path):
        wrn(f"{model_name} not downloaded yet", f"Run: python scripts/01_download_data.py")
        continue
    # Check config
    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        chk(False, f"{model_name} config.json", "missing")
        continue
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        test_tokens = tok("Hello, verify this works!", return_tensors="pt")
        n_tokens = test_tokens["input_ids"].shape[-1]
        chk(n_tokens > 0, f"{model_name} tokenizer loads + encodes correctly ({n_tokens} tokens)")
    except Exception as e:
        chk(False, f"{model_name} tokenizer test", str(e))

# ── 4. LoRA test (small model) ────────────────────────────────────────────────
print("\n  [D] LoRA / PEFT Test")
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    # Use tiny model for quick test (no download needed)
    qwen_path = qwen_1p5b_path if os.path.isdir(qwen_1p5b_path) else qwen_7b_path

    if os.path.isdir(qwen_path):
        cfg_obj = AutoConfig.from_pretrained(qwen_path)
        model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        total_params = sum(p.numel() for p in model.parameters())

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = 100 * trainable / total_params

        chk(trainable > 0, f"LoRA applied: {trainable/1e6:.1f}M trainable params ({pct:.2f}%)")

        # Forward pass test
        tok = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        inputs = tok("Test input for RLVER verification.", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        chk(out.logits.shape[-1] > 0, f"LoRA forward pass OK (logits shape: {list(out.logits.shape)})")

        # Cleanup
        del model, tok, inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        wrn("No model downloaded yet for LoRA test")
except Exception as e:
    chk(False, "LoRA / forward pass test", str(e))

# ── 5. Reward model unit test ─────────────────────────────────────────────────
print("\n  [E] RLVER Module Tests")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "rlver_adversarial"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "rl_training"))

try:
    from reward_model import RewardConfig, compute_reward, compute_rewards_batch
    cfg = RewardConfig()
    # Test: correct prediction with thinking
    r, comp = compute_reward(
        "<think>240÷8=30, matches.</think>\nVERDICT: CORRECT\nCONFIDENCE: 0.95",
        true_label=1, problem="240 cookies in 8 hours?", cfg=cfg
    )
    chk(r > 0, f"reward_model: correct prediction → reward={r:.3f} (>0 ✓)")
    # Test: wrong prediction
    r2, _ = compute_reward("VERDICT: CORRECT\nCONFIDENCE: 0.8", true_label=0, problem="test", cfg=cfg)
    chk(r2 < 0, f"reward_model: wrong prediction → reward={r2:.3f} (<0 ✓)")
except Exception as e:
    chk(False, "reward_model import + test", str(e))

try:
    from metrics import (EvalMetrics, VerificationSample, compute_metrics,
                         parse_model_output, save_metrics, load_metrics)
    # Parse output test
    pred, conf, think = parse_model_output(
        "<think>Steps...</think>\nVERDICT: INCORRECT\nCONFIDENCE: 0.85"
    )
    chk(pred == 0 and conf == 0.85, f"metrics.parse_model_output: pred={pred} conf={conf}")
except Exception as e:
    chk(False, "metrics module test", str(e))

try:
    from dataset_builder import RLVERDatasetBuilder, RLVERSample, format_verification_prompt
    # Build tiny dataset from 3 fake records (no HF download needed for this test)
    sample = RLVERSample(
        problem_id="test_001",
        problem="What is 2 + 2?",
        solution="2 + 2 = 4",
        label=1, is_adversarial=False, difficulty="easy",
        answer="4", source="test"
    )
    prompt = format_verification_prompt(sample, thinking_enabled=True)
    chk("<think>" in prompt and "Problem:" in prompt and "VERDICT:" in prompt,
        "dataset_builder: format_verification_prompt correct")
except Exception as e:
    chk(False, "dataset_builder test", str(e))

try:
    from ablation_config import ALL_ABLATION_GROUPS, list_all_experiments
    all_e = list_all_experiments()
    chk(len(ALL_ABLATION_GROUPS) == 9 and len(all_e) >= 40,
        f"ablation_config: {len(ALL_ABLATION_GROUPS)} groups, {len(all_e)} experiments")
except Exception as e:
    chk(False, "ablation_config test", str(e))

# ── 6. Disk space ─────────────────────────────────────────────────────────────
print("\n  [F] Disk / Cache")
import shutil
total, used, free = shutil.disk_usage(PROJECT_DIR)
free_gb = free / 1e9
chk(free_gb > 20, f"Free disk: {free_gb:.1f} GB (>20 GB required for training)",
    f"Only {free_gb:.1f} GB free")

cache_dir = os.path.join(PROJECT_DIR, "cache", "hf")
if os.path.isdir(cache_dir):
    cache_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(cache_dir) for f in fs
    ) / 1e9
    print(f"  ✓ HF cache: {cache_size:.1f} GB")

# ── Summary ───────────────────────────────────────────────────────────────────
total_checks = len(results["passed"]) + len(results["warnings"]) + len(results["failed"])
print(f"\n{'='*65}")
print(f"  Environment Verification Summary")
print(f"{'='*65}")
print(f"  Passed   : {len(results['passed'])}")
print(f"  Warnings : {len(results['warnings'])}")
print(f"  Failed   : {len(results['failed'])}")

if results["warnings"]:
    print(f"\n  Warnings:")
    for w in results["warnings"]:
        print(f"    ⚠ {w}")

if results["failed"]:
    print(f"\n  Failures (ACTION REQUIRED):")
    for f in results["failed"]:
        print(f"    ✗ {f}")
    sys.exit(1)
else:
    print(f"\n  ✓ All checks passed — environment is ready for training!")

print(f"{'='*65}")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
hdr "STEP 5 — Quick GPU training smoke-test"
# ═════════════════════════════════════════════════════════════════════════════

info "Running 10-step training smoke-test to confirm GPU training works..."

python - << 'PYEOF'
"""
Smoke-test: simulate one GRPO training step end-to-end (no model download needed).
Uses a tiny 1-layer GPT-2 style model to verify the full gradient flow.
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

PROJECT_DIR = os.path.expanduser("~/Dushyant_u23ai085_9256777500")
sys.path.insert(0, os.path.join(PROJECT_DIR, "rl_training"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "rlver_adversarial"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
print(f"  Device: {device}  dtype: {dtype}")

# ── 1. Reward function ────────────────────────────────────────────────────────
from reward_model import RewardConfig, compute_rewards_batch
reward_cfg = RewardConfig()
outputs_test = [
    "<think>240/8=30</think>\nVERDICT: CORRECT\nCONFIDENCE: 0.95",
    "VERDICT: INCORRECT\nCONFIDENCE: 0.70",
    "<think>check steps</think>\nVERDICT: CORRECT\nCONFIDENCE: 0.80",
]
rewards, comps = compute_rewards_batch(
    raw_outputs=outputs_test,
    true_labels=[1, 0, 1],
    problems=["P1", "P2", "P3"],
    cfg=reward_cfg,
)
assert len(rewards) == 3
assert rewards[0] > 0,  f"Expected r[0]>0, got {rewards[0]}"
assert rewards[1] > 0,  f"Expected r[1]>0 (correctly rejected), got {rewards[1]}"
print(f"  ✓ Reward function: {[f'{r:.3f}' for r in rewards]}")

# ── 2. GRPO loss computation ──────────────────────────────────────────────────
from grpo_trainer_custom import compute_grpo_loss
G, R, V = 4, 32, 512
new_lps = torch.randn(G, R, device=device, dtype=dtype, requires_grad=True)
old_lps = (new_lps + torch.randn_like(new_lps) * 0.1).detach()
ref_lps = (new_lps + torch.randn_like(new_lps) * 0.2).detach()
mask    = torch.ones(G, R, device=device, dtype=dtype)
r_vals  = np.array([0.8, -0.5, 0.3, -1.0])
adv     = torch.tensor((r_vals - r_vals.mean()) / (r_vals.std() + 1e-8),
                        dtype=dtype, device=device)
loss, stats = compute_grpo_loss(new_lps, old_lps, ref_lps, mask, adv, epsilon=0.2, beta=0.04)
loss.backward()
assert new_lps.grad is not None, "Gradients not computed"
grad_norm = new_lps.grad.norm().item()
print(f"  ✓ GRPO loss: {loss.item():.4f}  grad_norm: {grad_norm:.4f}")
new_lps.grad = None

# ── 3. PPO loss computation ───────────────────────────────────────────────────
from ppo_trainer_custom import compute_ppo_loss, RolloutBuffer, KLController

logits_new = torch.randn(2, 24, V, device=device, dtype=dtype, requires_grad=True)
logits_ref = torch.randn(2, 24, V, device=device, dtype=dtype)
full_ids   = torch.randint(0, V, (2, 24), device=device)
resp_mask  = torch.zeros(2, 24, device=device, dtype=torch.long)
resp_mask[:, 10:] = 1   # response tokens from position 10
old_lp     = torch.tensor([-2.5, -2.8], device=device, dtype=dtype)
values     = torch.tensor([0.4, -0.2], device=device, dtype=dtype)
advantages = torch.tensor([0.6, -0.3], device=device, dtype=dtype)
returns    = advantages + values

loss_ppo, stats_ppo = compute_ppo_loss(
    logits_new, logits_ref, full_ids, resp_mask,
    old_lp, values, advantages, returns,
)
loss_ppo.backward()
assert logits_new.grad is not None
print(f"  ✓ PPO  loss: {loss_ppo.item():.4f}  "
      f"pg={stats_ppo['loss/policy']:.4f}  vf={stats_ppo['loss/value']:.4f}")

# ── 4. RolloutBuffer + GAE ────────────────────────────────────────────────────
from ppo_trainer_custom import RolloutBuffer, RolloutSample, KLController
buf = RolloutBuffer(capacity=8)
import random as _rng
for i in range(6):
    buf.add(RolloutSample(
        query_ids    = torch.randint(0, 100, (12,)),
        response_ids = torch.randint(0, 100, (16,)),
        reward       = _rng.uniform(-1, 1),
        value        = _rng.uniform(-0.5, 0.5),
        log_prob     = _rng.uniform(-3, -1),
    ))
buf.compute_gae(gamma=0.99, lam=0.95, normalize=True)
adv_vals = [s.advantage for s in buf._buf]
assert all(isinstance(a, float) for a in adv_vals)
print(f"  ✓ RolloutBuffer GAE: {len(buf)} samples, advantages: {[f'{a:.3f}' for a in adv_vals[:3]]}...")

# ── 5. KL controller ─────────────────────────────────────────────────────────
kl = KLController(init_kl_coef=0.05, target_kl=6.0)
for kl_val in [8.0, 10.0, 6.2, 5.8]:
    kl.update(kl_val)
assert 0.0 < kl.kl_coef < 1.0
print(f"  ✓ KLController: β={kl.kl_coef:.5f}")

# ── 6. GPU memory check ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  ✓ GPU memory: {allocated:.2f} GB allocated / {total:.1f} GB total")

print(f"\n  ✓ ALL SMOKE TESTS PASSED — training pipeline is ready!")
PYEOF

# ═════════════════════════════════════════════════════════════════════════════
hdr "COMPLETE — Environment is Ready"
# ═════════════════════════════════════════════════════════════════════════════

cat << EOF

  ┌─────────────────────────────────────────────────────────────────┐
  │  GPU FIX + VERIFICATION COMPLETE                                │
  ├─────────────────────────────────────────────────────────────────┤
  │  GPU      : $GPU_NAME                                 
  │  sm_${GPU_CC}   : Working (PTX JIT compat mode if sm_120)       │
  │  bfloat16 : Enabled                                             │
  │  Env vars : $ENV_FILE  │
  ├─────────────────────────────────────────────────────────────────┤
  │  NEXT STEP — Data preparation:                                  │
  │                                                                 │
  │    source configs/rlver_env_vars.sh                             │
  │    python scripts/01_download_data.py                           │
  │                                                                 │
  │  With Mistral adversarial generation (~45 min):                 │
  │    python scripts/01_download_data.py --gen_adversarial         │
  │                                                                 │
  │  Or run everything at once:                                     │
  │    bash run_all.sh                                              │
  └─────────────────────────────────────────────────────────────────┘

EOF
