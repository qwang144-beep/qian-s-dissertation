"""
Generate reconstructed lyrics with Qwen2.5 (substituting the paper's ChatGPT-4o).

Reads the JSONL manifest from build_prompts.py and writes one <track_id>.txt per
song into --out-dir, so the result folder can be fed straight into
lycon_stats.py for the Table 1 comparison.

Run on Bessemer/Stanage with a GPU. Example (single node, 1 GPU):
    python generate.py --manifest prompts.jsonl --out-dir ./qwen_reconstructions \\
        --model Qwen/Qwen2.5-7B-Instruct

Notes
-----
* The paper does not report decoding settings for GPT-4o. We expose --temperature
  / --max-new-tokens; document whatever you use. Lower temperature -> more
  repetitive, which will further depress unique-unigram counts vs. the paper.
* --resume skips tracks whose .txt already exists, so the job is restartable
  within your HPC wall-clock limit.
* For throughput on HPC, consider vLLM instead of transformers; the prompt/
  output contract (one .txt per track_id) is identical.
"""
import argparse
import json
import os


SYSTEM = ("You are a songwriter. Write only the lyrics, using clear section "
          "headers like (Verse 1), (Chorus). Do not add commentary.")


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit NF4 quantization (fits a 7B on a Colab T4, ~5-6GB)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = dict(torch_dtype="auto", device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
        load_kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    items = list(read_manifest(args.manifest))
    print(f"{len(items)} prompts; model={args.model}")

    for i, item in enumerate(items, 1):
        out_path = os.path.join(args.out_dir, f"{item['track_id']}.txt")
        if args.resume and os.path.exists(out_path):
            continue

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": item["prompt"]}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                           skip_special_tokens=True).strip()
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(reply)

        if i % 50 == 0:
            print(f"  {i}/{len(items)}")

    print("done.")


if __name__ == "__main__":
    main()
