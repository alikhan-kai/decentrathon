"""packed.py on the real model: padded vs packed forward/backward on real batches (T4).

    python tools/pack_check.py --out pack_check.json [--rows 8] [--batch 4]

Loads Qwen2.5-1.5B-Instruct in 4-bit NF4 (double quant, fp16 compute) + LoRA r=16 / alpha=32 on q_proj, v_proj as in the
task, applies the whole solution (all kernels), then runs the first `rows` rows of data/train.jsonl as batches of `batch`
twice: with packed.py unpatched (the padded transformers forward) and patched (packed). LoRA dropout is 0 here (random
masks of different sizes cannot be compared) and the B matrices are random, so every LoRA gradient is non-trivial.
A second padded run measures the run-to-run noise floor. Compared: loss, hidden states on the real positions, LoRA gradients.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_batches(tokenizer, rows, batch, max_length=512):
    data = [json.loads(line) for line in (REPO / "data/train.jsonl").read_text(encoding="utf-8").splitlines()[:rows]]
    texts = []
    for row in data:
        user = (row["instruction"] + ("\n" + row["input"] if row.get("input") else "")).strip()
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    batches = []
    for start in range(0, len(texts), batch):
        enc = tokenizer(texts[start:start + batch], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        batches.append((enc["input_ids"].cuda(), enc["attention_mask"].cuda()))
    return batches


def run(inner, ids, mask):
    """One forward + backward; the loss is a plain fp32 cross entropy over the real next-token targets."""
    inner.zero_grad(set_to_none=True)
    keep = mask.bool()
    with torch.autocast("cuda", dtype=torch.float16):
        out = inner.model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state
        packed = getattr(out, "packed_keep", None) is not None
        h = hidden[0] if packed else hidden[keep]
        target = torch.nn.functional.pad(ids, (0, 1), value=-100)[:, 1:]
        nxt = torch.nn.functional.pad(keep, (0, 1))[:, 1:]
        target = torch.where(keep & nxt, target, torch.full_like(target, -100))[keep]
        loss = torch.nn.functional.cross_entropy(inner.lm_head(h).float(), target, ignore_index=-100)
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in inner.named_parameters() if p.grad is not None}
    return {"packed": packed, "loss": loss.item(), "h": h.detach().float().cpu(), "grads": grads}


def compare(a, b):
    hd = (a["h"] - b["h"]).abs().max().item()
    scale = a["h"].abs().max().item()
    worst_rel, worst_cos, worst_name = 0.0, 1.0, None
    for name, ga in a["grads"].items():
        gb = b["grads"][name]
        rel = (ga - gb).abs().max().item() / max(ga.abs().max().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0).item()
        if rel > worst_rel:
            worst_rel, worst_name = rel, name
        worst_cos = min(worst_cos, cos)
    return {"loss_a": a["loss"], "loss_b": b["loss"], "loss_abs_diff": abs(a["loss"] - b["loss"]),
            "hidden_max_abs_diff": hd, "hidden_max_abs": scale, "hidden_rel_diff": hd / scale,
            "grad_params": len(a["grads"]), "grad_max_rel_diff": worst_rel, "grad_worst_param": worst_name,
            "grad_min_cosine": worst_cos}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="pack_check.json")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import solution
    from solution import packed

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    solution.apply()                    # all patches, before the model is built, as in the harness
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, dtype=torch.float16,
                                                 device_map={"": 0})
    model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                       target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    torch.manual_seed(0)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.02)
    model.train()
    inner = model.get_base_model()

    report = {"batches": [], "packed_enabled": packed.ENABLED}
    for ids, mask in build_batches(tokenizer, args.rows, args.batch):
        lens = mask.sum(1).tolist()
        packed.unpatch()
        padded1 = run(inner, ids, mask)
        padded2 = run(inner, ids, mask)
        packed.patch()
        pk = run(inner, ids, mask)
        entry = {"lens": lens, "T": ids.shape[1], "R_over_P": sum(lens) / (len(lens) * ids.shape[1]),
                 "packed_path_taken": pk["packed"], "padded_path_taken": not padded1["packed"],
                 "padded_vs_packed": compare(padded1, pk), "padded_vs_padded_noise": compare(padded1, padded2)}
        report["batches"].append(entry)
        print(json.dumps(entry, indent=1))
    report["torch_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    ok = all(b["packed_path_taken"] and b["padded_vs_packed"]["loss_abs_diff"] < 2e-3 * max(1.0, b["padded_vs_packed"]["loss_a"])
             and b["padded_vs_packed"]["grad_min_cosine"] > 0.999 and b["padded_vs_packed"]["hidden_rel_diff"] < 5e-2
             for b in report["batches"])
    report["verdict"] = "PASS" if ok else "CHECK"     # loose fp16 criteria; compare with the noise floor above
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("pack_check:", report["verdict"], "->", args.out)


if __name__ == "__main__":
    main()
