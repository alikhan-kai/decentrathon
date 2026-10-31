"""Bounded CPU parity in either local-diagnostic or pinned GPU-package environment.

Default mode REQUIRES datasets 4.8.5 and the native cached Qwen chat template.
No model loading, downloads, installs, CUDA calls or cache manipulation. The CPU
diagnostic flag explicitly relaxes only datasets version and missing template;
that output never counts as the pinned parity gate. Run with CUDA_VISIBLE_DEVICES=''.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time

from component_timing import Timings
from fixtures import no_target_rows, retained_rows, robust_rows
import overhead


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def authenticate_tokenizer(root, tokenizer, manifest):
    """Manifest must be supplied independently by the parent, never self-derived.

    JSON: model, revision, files_sha256 (complete tokenizer asset inventory),
    template_sha256 (loaded native template, UTF-8). No network/source reads.
    """
    assert manifest["model"] == MODEL and manifest["revision"] == REVISION
    expected = manifest["files_sha256"]
    assert isinstance(expected, dict) and "tokenizer.json" in expected
    assert "tokenizer_config.json" in expected
    def asset(path):
        return (path.name.startswith(("tokenizer", "chat_template", "special_tokens"))
                or path.name in ("vocab.json", "merges.txt", "added_tokens.json")
                or "chat_templates" in path.parts)
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*")) if p.is_file() and asset(p.relative_to(root))}
    assert actual == expected, "tokenizer assets differ from parent trusted snapshot manifest"
    assert type(tokenizer.chat_template) is str and tokenizer.chat_template
    digest = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    assert digest == manifest["template_sha256"], "loaded template differs from trusted native template"
    return actual, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local complete Qwen tokenizer directory")
    parser.add_argument("--public-rows", required=True, help="Public task train.jsonl, exactly 256 Alpaca rows")
    parser.add_argument("--public-eval", required=True, help="Public eval_public.jsonl, exactly 256 Alpaca rows")
    parser.add_argument("--cpu-diagnostic", action="store_true")
    parser.add_argument("--trusted-tokenizer-manifest", help="Parent-supplied trusted native snapshot JSON; required in strict mode")
    parser.add_argument("--seconds", type=int, default=180, choices=range(30,301))
    args = parser.parse_args()
    if not args.cpu_diagnostic:
        assert args.trusted_tokenizer_manifest, "strict gate requires a parent-supplied trusted tokenizer manifest"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run this text-only gate with CUDA_VISIBLE_DEVICES=''"
    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    def timeout(signum, frame):
        raise TimeoutError("bounded parity gate exceeded its time limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    pins = {name: importlib.metadata.version(name) for name in
            ("datasets", "soup-cli", "transformers", "trl", "torch", "peft")}
    assert pins["datasets"] == ("5.1.0" if args.cpu_diagnostic else "4.8.5"), pins
    assert pins["soup-cli"] == "0.77.0" and pins["transformers"] == "5.16.1" and pins["trl"] == "0.29.1", pins
    import torch
    from datasets import Dataset
    from soup_cli.config.schema import DataConfig, TrainingConfig
    from soup_cli.data.formats import _convert_alpaca
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.monitoring.rewind_log import dataset_fingerprint
    from soup_cli.trainer import sft
    from transformers import AutoTokenizer
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    root = Path(args.tokenizer)
    assert root.is_dir(), "tokenizer must already be locally cached"
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    assert type(tokenizer) is Qwen2Tokenizer and tokenizer.vocab_size == 151643
    template_present = bool(tokenizer.chat_template)
    native_template = False
    trusted_manifest = None
    if args.trusted_tokenizer_manifest:
        trusted_manifest = json.loads(Path(args.trusted_tokenizer_manifest).read_text())
        authenticate_tokenizer(root, tokenizer, trusted_manifest)
        native_template = True
    cfg = DataConfig(train=args.public_rows, format="alpaca", max_length=512,
                     chat_template="qwen2.5" if not template_present else None)
    formatter = build_format_row(tokenizer, cfg, training_cfg=TrainingConfig())
    assert overhead._supported_formatter(formatter, 512)
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    raw = [json.loads(line) for line in Path(args.public_rows).read_text().splitlines() if line.strip()]
    assert len(raw) == 256, "this gate uses the complete 256-row public workload"
    public = [_convert_alpaca(row) for row in raw]
    raw_eval = [json.loads(line) for line in Path(args.public_eval).read_text().splitlines() if line.strip()]
    assert len(raw_eval) == 256
    public_eval = [_convert_alpaca(row) for row in raw_eval]
    original = sft._map_text_sft_rows
    candidate = overhead._make_map(original)
    result = {"mode": "cpu_diagnostic_NOT_PINNED_GATE" if args.cpu_diagnostic else "pinned_text_parity_gate",
              "versions": pins, "native_template": native_template,
              "template_provenance": "parent_trusted_snapshot_verified" if native_template else "unverified_cpu_diagnostic",
              "trusted_manifest_sha256": hashlib.sha256(Path(args.trusted_tokenizer_manifest).read_bytes()).hexdigest() if trusted_manifest else None,
              "model_revision": REVISION if native_template else None,
              "tokenizer_files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(root.glob("*")) if p.is_file() and
                                        (p.name.startswith("tokenizer") or p.name.startswith("chat_template"))},
              "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "public_source_sha256": hashlib.sha256(Path(args.public_rows).read_bytes()).hexdigest(),
              "public_eval_source_sha256": hashlib.sha256(Path(args.public_eval).read_bytes()).hexdigest(),
              "cases": {}, "timings": [], "remaining": ["actual model placement/quantization/hooks",
              "actual training Rewind ids/counts/losses", "official same-session T4 full runs"]}
    for name, rows in (("public", public), ("public_eval", public_eval),
                       ("robust", robust_rows()), ("retained", retained_rows())):
        assert overhead._supported_schema(Dataset.from_list(rows)), name
        before = copy.deepcopy(rows)
        rewind_fp = dataset_fingerprint(rows)
        a = original(rows, format_row=formatter, split="train", max_length=512)
        b = candidate(rows, format_row=formatter, split="train", max_length=512)
        assert a.to_list() == b.to_list(), name
        assert a.features == b.features and a.column_names == b.column_names and a.format == b.format, name
        assert rows == before and dataset_fingerprint(rows) == rewind_fp, name
        assert a.cache_files == b.cache_files == [], name
        # Keep Soup's assistant-mask postprocessing and the actual TRL collator.
        aa, bb = sft._ensure_assistant_masks(a), sft._ensure_assistant_masks(b)
        assert aa.features == bb.features and aa.to_list() == bb.to_list(), name
        # Independent permutations test alignment, not a claim about Trainer's RNG.
        orders = [list(range(len(a))), list(reversed(range(len(a))))]
        batches = 0
        counts = []
        for order in orders:
            for start in range(0, len(order), 4):
                indices = order[start:start+4]
                ba, bc = collator([aa[i] for i in indices]), collator([bb[i] for i in indices])
                assert ba.keys() == bc.keys()
                assert all(torch.equal(ba[key], bc[key]) for key in ba), (name, indices)
                counts.extend((ba["labels"][:,1:] != -100).sum(1).tolist())
                batches += 1
        result["cases"][name] = {"rows": len(rows), "output_sha256": fingerprint(a.to_list()),
                                  "features": a.features.to_dict(), "columns": a.column_names,
                                  "rewind_input_fingerprint": rewind_fp,
                                  "collated_batches": batches, "causal_counts_sha256": fingerprint(counts)}
    errors = []
    for fn in (original, candidate):
        rows = no_target_rows()
        before = copy.deepcopy(rows)
        try:
            fn(rows, format_row=formatter, split="validation", max_length=512)
        except ValueError as exc:
            errors.append((type(exc).__name__, str(exc), type(exc.__cause__).__name__, str(exc.__cause__)))
        else:
            raise AssertionError("expected no-causal-target error")
        assert rows == before
    assert errors[0] == errors[1]
    result["causal_error"] = errors[0]
    # Fresh in-memory Arrow construction per sample, same formatter and rows;
    # no filesystem cache reads/writes/deletions by this diagnostic.
    for name, fn in (("original", original), ("normalized", candidate),
                     ("normalized", candidate), ("original", original)):
        timer = Timings()
        timer.map_components(overhead)
        try:
            with timer.span("map_total"):
                ds = fn(public, format_row=formatter, split="train", max_length=512)
        finally:
            timer.restore()
        assert len(ds) == 256 and ds.cache_files == []
        result["timings"].append({"variant": name, "components_inclusive": timer.records})
    result["cuda_initialized"] = torch.cuda.is_initialized()
    assert not result["cuda_initialized"]
    result["passed"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False))
    signal.alarm(0)


if __name__ == "__main__":
    main()
