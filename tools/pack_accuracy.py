"""Accuracy of the packed forward/backward on the real model, against a padded fp32 reference (T4).

    python tools/pack_accuracy.py --out pack_accuracy.json [--batch 4]

Three real batches of data/train.jsonl (one has an odd number of real tokens). LoRA gradients in three variants:
  (a) padded fp32  - reference (compute dtype fp32, no autocast, packed.py unpatched)
  (b) padded fp16  - the training setup without packing
  (c) packed fp16  - packed.py patched
(b) and (c) are compared with (a): relative L2 error and cosine, per group (q_proj / v_proj x lora_A / lora_B; every
layer's tensor concatenated). The same LoRA weights (B random, dropout 0) are used in all three. If the error of (c) is
larger than the error of (b), the hidden states after layers 0, 9, 18, 27 and the final norm are compared too, to see
where the extra error appears. Nothing is "fixed" here: it only measures.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HOOKED_LAYERS = (0, 9, 18, 27)


def build_batches(tokenizer, batch, max_length=512):
    rows = [json.loads(line) for line in (REPO / "data/train.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = []
    for row in rows[:48]:
        user = (row["instruction"] + ("\n" + row["input"] if row.get("input") else "")).strip()
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    groups = []
    for start in range(0, len(texts) - batch + 1, batch):
        enc = tokenizer(texts[start:start + batch], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        groups.append((enc["input_ids"], enc["attention_mask"]))
    chosen = groups[:2]
    if not any(int(m.sum()) % 2 for _, m in chosen):          # make sure one batch has an odd number of real tokens
        chosen.append(next(g for g in groups[2:] if int(g[1].sum()) % 2))
    else:
        chosen.append(groups[2])
    return [(i.cuda(), m.cuda()) for i, m in chosen]


def run(inner, ids, mask, fp16):
    """Forward + backward with a plain fp32 cross entropy on the real next-token targets; returns LoRA grads and the
    hidden states (real tokens only, row-major) after the hooked layers."""
    inner.zero_grad(set_to_none=True)
    keep = mask.bool()
    captured = {}
    handles = []
    for index in HOOKED_LAYERS:
        def hook(_module, _inputs, output, index=index):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index] = (tensor[0] if tensor.shape[0] == 1 and keep.shape[0] != 1 else tensor[keep]).detach().float().cpu()
        handles.append(inner.model.layers[index].register_forward_hook(hook))
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        out = inner.model(input_ids=ids, attention_mask=mask)
        packed = getattr(out, "packed_keep", None) is not None
        hidden = out.last_hidden_state
        h = hidden[0] if packed else hidden[keep]
        target = torch.nn.functional.pad(ids, (0, 1), value=-100)[:, 1:]
        nxt = torch.nn.functional.pad(keep, (0, 1))[:, 1:]
        target = torch.where(keep & nxt, target, torch.full_like(target, -100))[keep]
        loss = torch.nn.functional.cross_entropy(inner.lm_head(h).float(), target, ignore_index=-100)
    loss.backward()
    for handle in handles:
        handle.remove()
    captured["final"] = h.detach().float().cpu()
    grads = {n: p.grad.detach().float().cpu().clone() for n, p in inner.named_parameters() if p.grad is not None}
    return {"packed": packed, "loss": loss.item(), "grads": grads, "hidden": captured}


def group_of(name):
    proj = "q_proj" if ".q_proj." in name else "v_proj"
    return f"{proj}.{'lora_A' if 'lora_A' in name else 'lora_B'}"


def grouped(grads):
    groups = {}
    for name in sorted(grads):
        groups.setdefault(group_of(name), []).append(grads[name].flatten())
    return {k: torch.cat(v) for k, v in groups.items()}


def error(ref, other):
    ref_g, other_g = grouped(ref["grads"]), grouped(other["grads"])
    return {k: {"rel_l2": ((other_g[k] - ref_g[k]).norm() / ref_g[k].norm()).item(),
                "cosine": torch.nn.functional.cosine_similarity(other_g[k], ref_g[k], dim=0).item()} for k in ref_g}


def hidden_error(ref, other):
    return {str(k): ((other["hidden"][k] - ref["hidden"][k]).norm() / ref["hidden"][k].norm()).item()
            for k in ref["hidden"]}


def load_model(dtype, lora_state):
    import peft
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, dtype=dtype, device_map={"": 0})
    torch.manual_seed(0)
    model = peft.get_peft_model(model, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                       target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    if lora_state is None:
        for name, p in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.02)
        lora_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}
    else:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.copy_(lora_state[name].to(p.dtype))
    model.train()
    return model, lora_state


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="pack_accuracy.json")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    import solution
    from solution import packed

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "right"
    solution.apply()
    batches = build_batches(tokenizer, args.batch)
    info = [{"lens": m.sum(1).tolist(), "real_tokens": int(m.sum()), "T": m.shape[1]} for _, m in batches]
    print("batches:", info)

    model, lora_state = load_model(torch.float32, None)             # (a)
    packed.unpatch()
    ref = [run(model.get_base_model(), i, m, fp16=False) for i, m in batches]
    del model
    torch.cuda.empty_cache()

    model, _ = load_model(torch.float16, lora_state)                # (b), (c)
    inner = model.get_base_model()
    report = {"batches": []}
    for k, (ids, mask) in enumerate(batches):
        packed.unpatch()
        b = run(inner, ids, mask, fp16=True)
        packed.patch()
        c = run(inner, ids, mask, fp16=True)
        eb, ec = error(ref[k], b), error(ref[k], c)
        entry = {**info[k], "padded_path_b": not b["packed"], "packed_path_c": c["packed"],
                 "loss_a": ref[k]["loss"], "loss_b": b["loss"], "loss_c": c["loss"],
                 "grad_error_b_padded_fp16": eb, "grad_error_c_packed_fp16": ec,
                 "c_not_worse_than_b": {g: ec[g]["rel_l2"] <= eb[g]["rel_l2"] for g in eb},
                 "hidden_rel_l2_b": hidden_error(ref[k], b), "hidden_rel_l2_c": hidden_error(ref[k], c)}
        report["batches"].append(entry)
        print(json.dumps(entry, indent=1))
    report["c_not_worse_than_b_everywhere"] = all(all(e["c_not_worse_than_b"].values()) for e in report["batches"])
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("pack_accuracy: c not worse than b everywhere:", report["c_not_worse_than_b_everywhere"])


if __name__ == "__main__":
    main()
