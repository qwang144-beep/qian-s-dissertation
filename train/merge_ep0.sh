#!/bin/bash
#SBATCH --job-name=mergeep0
#SBATCH --partition=sheffield
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=mergeep0_%j.out
PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
cd /users/smp23qw/dissertation/lycon
$PY - <<'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, "lora_out/epoch0")
m = m.merge_and_unload()
out = "/mnt/parscratch/users/smp23qw/qwen_metrical_sft_ep0"
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained(out)
print("merged ->", out)
PYEOF
