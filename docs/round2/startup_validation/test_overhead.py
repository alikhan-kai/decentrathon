"""CPU-only tests with real libraries; no fake dependencies or TRL modules."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import FunctionType, SimpleNamespace
import tempfile
import hashlib
from unittest.mock import patch

import torch
from datasets import Dataset
from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
from soup_cli.data.sft_format import build_format_row
from soup_cli.trainer import sft
from soup_cli.utils.gpu import resolve_device_map
from transformers import AutoTokenizer

import overhead as candidate
from fixtures import chat, no_target_rows, retained_rows, robust_rows

HERE = Path(__file__).resolve().parent
TOKENIZER = os.environ.get("STARTUP_TOKENIZER", "/tmp/rotary-r1-evidence/candidate/runs/base_1/adapter")


class MapParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        cls.native_template = bool(cls.tokenizer.chat_template)
        cls.data = DataConfig(train="unused", format="alpaca", max_length=512,
                              chat_template=None if cls.native_template else "qwen2.5")
        cls.formatter = staticmethod(build_format_row(cls.tokenizer, cls.data,
                                                     training_cfg=TrainingConfig()))
        cls.original = staticmethod(sft._map_text_sft_rows)
        cls.candidate = staticmethod(candidate._make_map(sft._map_text_sft_rows))

    def compare(self, rows, formatter=None, length=512):
        before = copy.deepcopy(rows)
        formatter = formatter or self.formatter
        kw = dict(format_row=formatter, split="validation", max_length=length)
        a = self.original(rows, **kw)
        b = self.candidate(rows, **kw)
        self.assertEqual(a.column_names, b.column_names)
        self.assertEqual(a.features, b.features)
        self.assertEqual(a.to_list(), b.to_list())
        self.assertEqual(a.format, b.format)
        self.assertEqual(a.cache_files, b.cache_files)
        self.assertEqual(rows, before)
        return a, b

    def test_real_soup_qwen_unicode_multiturn_truncation_order(self):
        a, _ = self.compare(robust_rows())
        self.assertEqual(len(a), 9)
        self.assertEqual(len(a[6]["input_ids"]), 512)

    def test_retained_columns_missing_keys_nulls_merge_features_order(self):
        a, _ = self.compare(retained_rows())
        self.assertEqual(a.column_names, ["row_id", "extra", "score", "flag", "empty",
                                         "input_ids", "labels", "attention_mask"])
        self.assertEqual(a[1]["extra"], None)
        self.assertEqual(a[1]["empty"], None)
        self.assertNotIn("later_only", a.column_names)

    def test_eligible_path_retains_initial_normalization_and_never_calls_map(self):
        calls = []
        original = Dataset.from_list.__func__
        def record(cls, data, *args, **kwargs):
            calls.append(copy.deepcopy(data))
            return original(cls, data, *args, **kwargs)
        with patch.object(Dataset, "from_list", classmethod(record)), patch.object(
            Dataset, "map", side_effect=AssertionError("eligible path called Dataset.map")
        ):
            out = self.candidate(retained_rows(), format_row=self.formatter,
                                 split="train", max_length=512)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(out[1]["extra"])

    def assert_fallback_before_formatting(self, formatter, calls):
        seen = []
        def original(rows, **kwargs):
            self.assertEqual(calls, [], "formatter ran before fallback decision")
            seen.append(True)
            return self.original(rows, **kwargs)
        with patch.object(candidate, "_map_normalized", side_effect=AssertionError("must fall back")):
            out = candidate._make_map(original)([chat()], format_row=formatter,
                                                split="train", max_length=512)
        self.assertEqual(seen, [True])
        self.assertEqual(len(calls), 1, "fallback must not replay formatting")
        self.assertEqual(out[0]["audit_extra"], 17)

    def test_changed_global_helper_falls_back_before_any_row(self):
        from soup_cli.data import sft_format
        actual = sft_format.build_assistant_only_labels
        calls = []
        def changed(*args, **kwargs):
            calls.append(True)
            return actual(*args, **kwargs) | {"audit_extra": 17}
        for reference in (candidate._formatter_reference, None):
            calls.clear()
            with patch.object(candidate, "_formatter_reference", reference), patch.object(
                sft_format, "build_assistant_only_labels", changed
            ):
                self.assertFalse(candidate._supported_formatter(self.formatter, 512))
                self.assert_fallback_before_formatting(self.formatter, calls)

    def test_same_code_different_globals_falls_back_before_any_row(self):
        from soup_cli.data import sft_format
        actual = sft_format.build_assistant_only_labels
        calls = []
        def changed(*args, **kwargs):
            calls.append(True)
            return actual(*args, **kwargs) | {"audit_extra": 17}
        inner = candidate._closure(self.formatter)["inner"]
        other = FunctionType(inner.__code__, inner.__globals__ | {"build_assistant_only_labels": changed},
                             inner.__name__, inner.__defaults__, inner.__closure__)
        formatter = sft_format._wrap_dropping_arrow_nulls(other)
        self.assertFalse(candidate._supported_formatter(formatter, 512))
        self.assert_fallback_before_formatting(formatter, calls)
        self.assertIs(sft_format.build_assistant_only_labels, actual)
        copied_wrapper = FunctionType(self.formatter.__code__, dict(self.formatter.__globals__),
                                      self.formatter.__name__, self.formatter.__defaults__,
                                      self.formatter.__closure__)
        self.assertFalse(candidate._supported_formatter(copied_wrapper, 512))

    def test_mutated_helper_code_and_factory_rejected_before_rows(self):
        from soup_cli.data import sft_format, loss_mask
        self.assertTrue(candidate._supported_formatter(self.formatter, 512))
        def changed(*args, **kwargs):
            raise AssertionError("modified function must not run")
        for fn in (loss_mask.build_assistant_only_labels, loss_mask._tokenize_only,
                   sft_format._build_assistant_only_format_row, sft_format._wrap_dropping_arrow_nulls):
            code = fn.__code__
            try:
                fn.__code__ = changed.__code__
                self.assertFalse(candidate._supported_formatter(self.formatter, 512))
                with patch.object(candidate, "_formatter_reference", None):
                    self.assertFalse(candidate._supported_formatter(self.formatter, 512))
            finally:
                fn.__code__ = code
        self.assertTrue(candidate._supported_formatter(self.formatter, 512))
        with patch.object(sft_format, "_build_assistant_only_format_row",
                          FunctionType(sft_format._build_assistant_only_format_row.__code__,
                                       dict(vars(sft_format)), argdefs=(False, False))):
            self.assertFalse(candidate._supported_formatter(self.formatter, 512))


    def test_original_causal_error_type_message_and_cause(self):
        errors = []
        rows = no_target_rows()
        before = copy.deepcopy(rows)
        for fn in (self.original, self.candidate):
            with self.assertRaises(ValueError) as raised:
                fn(rows, format_row=self.formatter, split="validation", max_length=512)
            exc = raised.exception
            errors.append((type(exc), str(exc), type(exc.__cause__), str(exc.__cause__)))
        self.assertEqual(*errors)
        self.assertTrue(errors[0][1].startswith("validation row 2:"))
        self.assertEqual(rows, before)

    def test_sparse_nested_key_uses_original_before_formatter(self):
        rows = [chat()]
        rows[0]["messages"][1]["name"] = "assistant_name"
        seen = []
        def formatter(example):
            seen.append(copy.deepcopy(dict(example)))
            return {"text": str("name" in example["messages"][0])}
        a, _ = self.compare(rows, formatter)
        self.assertEqual(a[0]["text"], "True")
        self.assertEqual(len(seen), 2)  # once per independent baseline/candidate run

    def test_custom_mutating_formatter_fallback_once_and_input_unchanged(self):
        rows = retained_rows()
        seen = []
        def formatter(example):
            seen.append(example.pop("extra"))
            example["messages"][0]["content"] = "mutated normalized row"
            return {"text": str(len(seen))}
        before = copy.deepcopy(rows)
        self.candidate(rows, format_row=formatter, split="train", max_length=512)
        self.assertEqual(len(seen), len(rows))
        self.assertEqual(rows, before)

    def test_missing_messages_preserves_original_error_before_formatter(self):
        calls = []
        def f(row):
            calls.append(row)
            return {"text": "unused"}
        errors = []
        for fn in (self.original, self.candidate):
            with self.assertRaises(ValueError) as raised:
                fn([{"extra": "x"}], format_row=f, split="train", max_length=512)
            errors.append(str(raised.exception))
        self.assertEqual(*errors)
        self.assertEqual(calls, [])

    def test_empty_original_and_overwrite_schema_fallback(self):
        errors = []
        for fn in (self.original, self.candidate):
            with self.assertRaises(ValueError) as raised:
                fn([], format_row=self.formatter, split="train", max_length=512)
            errors.append(str(raised.exception))
        self.assertEqual(*errors)
        rows = [chat()]
        rows[0]["input_ids"] = [1, 2]
        self.assertFalse(candidate._supported_schema(Dataset.from_list(rows)))
        self.compare(rows)

    def test_unsupported_formatter_modes_and_max_length(self):
        for updates in (dict(train_on_responses_only=False), dict(mask_history=True)):
            cfg = self.data.model_copy(update=updates)
            f = build_format_row(self.tokenizer, cfg, training_cfg=TrainingConfig())
            self.assertFalse(candidate._supported_formatter(f, 512))
            self.compare(robust_rows()[:3], f)
        self.assertFalse(candidate._supported_formatter(self.formatter, 128))

    def test_unsupported_schema_guard_precedes_any_formatter_call(self):
        rows = [chat()]
        rows[0]["metadata"] = {"nested": [1, 2]}
        self.assertFalse(candidate._supported_schema(Dataset.from_list(rows)))
        self.compare(rows)

    def test_core_no_replay_after_stateful_failure(self):
        calls = []
        def f(example):
            calls.append(example["messages"][0]["content"])
            example["messages"][0]["content"] = "changed"
            raise RuntimeError("formatter consumed state")
        rows = [chat()]
        before = copy.deepcopy(rows)
        with self.assertRaisesRegex(RuntimeError, "consumed state"):
            candidate._map_normalized(Dataset.from_list(rows), format_row=f,
                                      split="train", max_length=512)
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows, before)


class DevicePolicy(unittest.TestCase):
    def test_cpu_and_all_other_devices_use_real_original(self):
        token = candidate._fixed_setup.set(True)
        try:
            for device in ("cpu", torch.device("cpu"), "cuda", torch.device("cuda:0"),
                           "cuda:1", torch.device("cuda:1"), "mps", "xpu", "cuda:banana", None):
                self.assertEqual(candidate._resolve(resolve_device_map, device), resolve_device_map(device))
        finally:
            candidate._fixed_setup.reset(token)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_error_is_not_suppressed_or_probe_called(self):
        error = ValueError("original error")
        def original(_):
            raise error
        def never(_):
            self.fail("eligibility must follow original")
        with self.assertRaises(ValueError) as raised:
            candidate._resolve(original, "cuda", never)
        self.assertIs(raised.exception, error)

    def test_only_auto_result_replaced(self):
        self.assertEqual(candidate._resolve(lambda _: "auto", "cuda", lambda _: True), {"": 0})
        for result in ("cpu", {"": 2}, {"a": "cpu", "b": 0}):
            self.assertIs(candidate._resolve(lambda _: result, "cuda", lambda _: True), result)

    def test_real_distributed_resolver_and_malformed_environments(self):
        for env in ({"LOCAL_RANK": "2", "WORLD_SIZE": "4"},
                    {"LOCAL_RANK": "bad", "WORLD_SIZE": "bad"},
                    {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}):
            with patch.dict(os.environ, env):
                self.assertEqual(candidate._resolve(resolve_device_map, "cuda"), resolve_device_map("cuda"))

    def test_cuda_policy_facts_without_fake_cuda(self):
        facts = dict(cuda_build=True, available=True, visible_count=1, current_device=0,
                     distributed=False, env={}, free_bytes=10*1024**3, total_bytes=15*1024**3)
        self.assertTrue(candidate._eligible_facts(**facts))
        for update in (dict(cuda_build=False), dict(available=False), dict(visible_count=2),
                       dict(current_device=1), dict(distributed=True), dict(free_bytes=0),
                       dict(total_bytes=8*1024**3), dict(env={"WORLD_SIZE": "2"}),
                       dict(env={"LOCAL_RANK": "0"}), dict(env={"RANK": "bad"})):
            self.assertFalse(candidate._eligible_facts(**(facts | update)))

    def test_fixed_model_context_and_exception_cleanup(self):
        cfg = SoupConfig(base=candidate.SUPPORTED_MODEL, data=DataConfig(train="unused",format="alpaca",max_length=512))
        self.assertTrue(candidate._supported_setup(cfg, cfg.training))
        for updates in (dict(base="other/Qwen2.5-1.5B-Instruct"), dict(base="Qwen/Qwen2.5-7B-Instruct")):
            other = cfg.model_copy(update=updates)
            self.assertFalse(candidate._supported_setup(other, other.training))
        def setup(self, cfg, tcfg):
            self.assertTrue(candidate._fixed_setup.get())
            raise ValueError("setup error")
        with self.assertRaisesRegex(ValueError, "setup error"):
            candidate._make_setup(setup)(self, cfg, cfg.training)
        self.assertFalse(candidate._fixed_setup.get())

    def test_other_quantization_and_distributed_setup_options(self):
        cfg = SoupConfig(base=candidate.SUPPORTED_MODEL, data=DataConfig(train="unused",format="alpaca",max_length=512))
        other_training = cfg.training.model_copy(update={"quantization": "8bit"})
        other = cfg.model_copy(update={"training": other_training})
        self.assertFalse(candidate._supported_setup(other, other.training))
        for key in ("fsdp_config", "deepspeed_config", "_trust_remote_code"):
            # Actual Soup wrapper, no fake transformers/peft/module objects.
            wrapper = sft.SFTTrainerWrapper(cfg)
            setattr(wrapper, key, True)
            def setup(instance, cfg, tcfg):
                return candidate._fixed_setup.get()
            self.assertFalse(candidate._make_setup(setup)(wrapper, cfg, cfg.training))


class GateContracts(unittest.TestCase):
    def test_all_argparse_profile_prefixes_rejected(self):
        from setup_probe import guard_harness_args
        for end in range(3, len("--profile") + 1):
            option = "--profile"[:end]
            for args in ([option, "/tmp/log"], [option + "=/tmp/log"]):
                with self.assertRaisesRegex(AssertionError, "truncates"):
                    guard_harness_args(["--solution", *args])
        guard_harness_args(["--solution", "--config", "run.yaml"])

    def test_protected_epoch_budget_is_accepted_without_dataloader(self):
        from setup_probe import capture_training
        args = SimpleNamespace(max_steps=-1, num_train_epochs=1,
            per_device_train_batch_size=4, gradient_accumulation_steps=1,
            world_size=1, dataloader_drop_last=False, auto_find_batch_size=False)
        trainer = SimpleNamespace(state=SimpleNamespace(global_step=0), args=args,
                                  train_dataset=list(range(256)), _train_batch_size=4)
        def finish(t):
            t.state.global_step=64
        state={}
        capture_training(finish,state)(trainer)
        self.assertTrue(state["training_completed"])
        for name,value in (("num_train_epochs",.5),("per_device_train_batch_size",8),
                           ("gradient_accumulation_steps",2),("world_size",2),
                           ("dataloader_drop_last",True),("auto_find_batch_size",True)):
            old=getattr(args,name);setattr(args,name,value);trainer.state.global_step=0
            with self.assertRaises(AssertionError):capture_training(finish,{})(trainer)
            setattr(args,name,old)

    def test_training_requires_successful_64_steps(self):
        from setup_probe import capture_training
        for steps in (0, 1, 63, 65):
            state = {}
            with self.assertRaises(AssertionError):
                capture_training(lambda trainer: setattr(trainer.state, "global_step", steps), state)(SimpleNamespace(state=SimpleNamespace(global_step=0), args=SimpleNamespace(max_steps=64)))
            self.assertNotIn("training_completed", state)
        state = {}
        def successful(trainer):
            trainer.state.global_step = 64
            return "done"
        result = capture_training(successful, state)(SimpleNamespace(state=SimpleNamespace(global_step=0), args=SimpleNamespace(max_steps=64)))
        self.assertEqual(result, "done")
        self.assertTrue(state["training_completed"])
        self.assertEqual(state["training_calls"], 1)
        def fail(trainer):
            raise RuntimeError("training failed")
        state = {}
        with self.assertRaises(RuntimeError):
            capture_training(fail, state)(SimpleNamespace(state=SimpleNamespace(global_step=0), args=SimpleNamespace(max_steps=64)))
        self.assertEqual(state, {})
        for start, budget, resume in ((63, 64, None), (0, 20, None), (0, 64, "checkpoint")):
            with self.assertRaises(AssertionError):
                capture_training(successful, {})(SimpleNamespace(state=SimpleNamespace(global_step=start), args=SimpleNamespace(max_steps=budget)), resume)

    def test_complete_ablation_matrix_and_placement(self):
        from compare_setup import compare
        def record(variant):
            fast = variant in ("map", "both")
            device = variant in ("device", "both")
            return dict(variant=variant, setup_completed=True, full_run_completed=True,
                        training_completed=True, training_calls=1, exit_status=0, completed_steps=64,
                        hook_evidence="counts_and_qualified_type_only", hooks=[], quantization_config={},
                        tensors=[{"name": "layer.weight", "device": "cuda:0"}], hf_device_map={"": 0},
                        components_inclusive={"infer_auto_device_map": {"calls": int(not device)},
                                              "balanced_memory": {"calls": int(not device)},
                                              "soup_device_resolver": {"calls": 1},
                                              "normalized_map_core_inclusive": {"calls": 2 if fast else 0},
                                              "text_map_total": {"calls": 2}})
        for variant in ("device", "map", "both"):
            self.assertTrue(compare(record("control"), record(variant))["setup_metadata_equal"])
            for contaminated in ("control", variant):
                for name in ("infer_auto_device_map", "balanced_memory", "normalized_map_core_inclusive"):
                    a, b = record("control"), record(variant)
                    target = a if contaminated == "control" else b
                    value = target["components_inclusive"][name]["calls"]
                    target["components_inclusive"][name]["calls"] = 0 if value else 1
                    with self.assertRaises(AssertionError):
                        compare(a, b)
        a, b = record("control"), record("map")
        b["hf_device_map"] = {"layer": "cuda:0"}
        compare(a, b)
        b["hf_device_map"] = {"": "cpu"}
        with self.assertRaises(AssertionError):
            compare(a, b)
        for key, bad in (("completed_steps", 63), ("exit_status", 1), ("full_run_completed", False)):
            b = record("map") | {key: bad}
            with self.assertRaises(AssertionError):
                compare(a, b)

    def test_native_provenance_requires_exact_parent_manifest(self):
        from parity_gate import authenticate_tokenizer, MODEL, REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("synthetic tokenizer fixture")
            (root / "tokenizer_config.json").write_text("synthetic config fixture")
            template = "synthetic trusted template"
            manifest = dict(model=MODEL, revision=REVISION,
                            files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()},
                            template_sha256=hashlib.sha256(template.encode()).hexdigest())
            authenticate_tokenizer(root, SimpleNamespace(chat_template=template), manifest)
            for wrong in ("replacement fallback template", ""):
                with self.assertRaises(AssertionError):
                    authenticate_tokenizer(root, SimpleNamespace(chat_template=wrong), manifest)
            (root / "tokenizer_config.json").write_text("overridden config")
            with self.assertRaises(AssertionError):
                authenticate_tokenizer(root, SimpleNamespace(chat_template=template), manifest)


class ImportAndInstall(unittest.TestCase):
    def test_fresh_process_real_import_chain_all_variants(self):
        for variant in ("none", "device", "map", "both"):
            done = subprocess.run([sys.executable, "-B", str(HERE / "fresh_import.py"), variant],
                                  text=True, capture_output=True, timeout=60, check=True)
            result = json.loads(done.stdout.splitlines()[-1])
            self.assertFalse(result["cuda_initialized"])
            self.assertTrue(result["identities_preserved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
