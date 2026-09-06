# -*- coding: utf-8 -*-
"""
train_lora.py -- LoRA SFT with a plain PyTorch loop (no trl/datasets dependency)
================================================================================
Distils the fixed selector's behaviour into the weights: takes the JSONL
produced by build_sft_data.py and computes the loss on the assistant span only
(prompt tokens get label=-100).

Produces two outputs:
  --out-dir     LoRA adapters (one per epoch + final)
  --merged-dir  the merged full model -- usable directly as
                generate_metrical.py's --model

See run_lora.sh for usage.
"""
from __future__ import annotations
import argparse, json, math, os, random, time

import torch
from torch.utils.data import Dataset, DataLoader


def build_example(tok, rec, max_len):
    msgs = [{"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": rec["assistant"]}],
        tokenize=False)
    if not full_text.startswith(prompt_text):      # incompatible template: supervise the whole span and warn
        prompt_text = ""
    p_ids = tok(prompt_text, add_special_tokens=False).input_ids
    f_ids = tok(full_text, add_special_tokens=False).input_ids
    if len(f_ids) > max_len or len(f_ids) <= len(p_ids):
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


class SftData(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows, skipped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = build_example(tok, json.loads(line), max_len)
                if ex is None:
                    skipped += 1
                else:
                    self.rows.append(ex)
        print(f"[data] {path}: {len(self.rows)} examples "
              f"({skipped} skipped as overlong/empty)")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        n = L - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * n)
        lab.append(b["labels"] + [-100] * n)
        att.append([1] * len(b["input_ids"]) + [0] * n)
    t = torch.tensor
    return {"input_ids": t(ids), "labels": t(lab), "attention_mask": t(att)}


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tot, nb = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tot += model(**batch).loss.item()
        nb += 1
    model.train()
    return tot / max(1, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out-dir", default="lora_out")
    ap.add_argument("--merged-dir", default="qwen_metrical_sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_ds = SftData(a.train, tok, a.max_len)
    val_ds = SftData(a.val, tok, a.max_len) if a.val else None
    cl = lambda b: collate(b, tok.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=a.bsz, shuffle=True, collate_fn=cl)
    val_dl = DataLoader(val_ds, batch_size=a.bsz, collate_fn=cl) if val_ds else None

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha,
                      lora_dropout=a.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    device = "cuda"
    model.to(device)

    steps_per_epoch = math.ceil(len(train_dl) / a.accum)
    total_steps = steps_per_epoch * a.epochs
    warm = max(1, int(total_steps * a.warmup))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=a.lr, weight_decay=0.0)
    def lr_at(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    model.train()
    step = micro = 0
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    for ep in range(a.epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / a.accum
            loss.backward()
            micro += 1
            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"ep {ep} step {step}/{total_steps} "
                          f"loss {loss.item()*a.accum:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"{time.time()-t0:.0f}s", flush=True)
        if val_dl:
            print(f"[eval] epoch {ep}: val loss {eval_loss(model, val_dl, device):.4f}",
                  flush=True)
        model.save_pretrained(os.path.join(a.out_dir, f"epoch{ep}"))

    model.save_pretrained(os.path.join(a.out_dir, "final"))
    print("[merge] merging adapter into base weights ...", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(a.merged_dir, safe_serialization=True)
    tok.save_pretrained(a.merged_dir)
    print(f"[done] merged model -> {a.merged_dir}  "
          f"(pass this path to generate_metrical.py --model)")


if __name__ == "__main__":
    main()
