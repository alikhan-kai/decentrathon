"""Parent-only diagnostic launcher: same full harness workload, setup timers.

Requires an independently assembled checkout containing this candidate module
and initializer delta. This is NEVER part of the scored solution. It adds timers
around setup calls only and leaves the full original harness command running.
It does not stop after setup, shorten steps, invoke a second model forward, read
or alter caches, change environment variables, or alter harness files. Run both
control/candidate instrumented. Use normal uninstrumented harness repetitions
for performance claims. Only stdout diagnostic output is added.
"""
import argparse
import atexit
import importlib
import json
from pathlib import Path
import runpy
import sys

from component_timing import Timings



def guard_harness_args(arguments):
    # argparse accepts every unambiguous long-option prefix, with or without '='.
    # Reject even potentially ambiguous prefixes; never execute a truncated run.
    for argument in arguments:
        option = argument.split("=", 1)[0]
        assert not (option.startswith("--") and len(option) > 2
                    and "--profile".startswith(option)), "profile mode truncates the workload"
    assert "--solution" in arguments, "require solution-enabled full workload"


def capture_training(original, state):
    def train(trainer, *args, **kwargs):
        assert trainer.state.global_step == 0, "require a fresh full training run"
        if trainer.args.max_steps != 64:
            # The protected task uses one epoch over 256 rows, not max_steps.
            # Inspect metadata only; constructing a second dataloader could alter RNG.
            a = trainer.args
            assert a.max_steps <= 0 and a.num_train_epochs == 1
            assert len(trainer.train_dataset) == 256
            assert a.per_device_train_batch_size == 4 and a.gradient_accumulation_steps == 1
            assert a.world_size == 1 and not a.dataloader_drop_last
            assert a.auto_find_batch_size is False
            assert getattr(trainer, "_train_batch_size", 4) == 4

        resume = args[0] if args else kwargs.get("resume_from_checkpoint")
        assert resume is None or resume is False, "resumed training is not a full 64-step run"
        result = original(trainer, *args, **kwargs)
        steps = trainer.state.global_step
        state["completed_steps"] = steps
        state["training_calls"] = state.get("training_calls", 0) + 1
        assert type(steps) is int and steps == 64, "full workload must complete exactly 64 steps"
        state["training_completed"] = True
        return result
    return train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--variant", required=True, choices=("control", "device", "map", "both"))
    parser.add_argument("harness_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    harness_args = args.harness_args
    if harness_args[:1] == ["--"]:
        harness_args = harness_args[1:]
    guard_harness_args(harness_args)
    sys.path.insert(0, str(repo))
    solution = importlib.import_module("solution")
    candidate = importlib.import_module("solution.overhead")
    assert Path(solution.__file__).resolve().parent == repo / "solution"
    assert solution.MODULES == ["swiglu", "head_loss", "rmsnorm", "nf4", "lora", "rotary", "overhead"]
    candidate.ENABLED = args.variant != "control"
    candidate.DEVICE_MAP = args.variant in ("device", "both")
    candidate.FAST_MAP = args.variant in ("map", "both")
    timer = Timings()
    original = solution.apply
    state = {"variant": args.variant, "setup_completed": False,
             "training_completed": False, "full_run_completed": False, "exit_status": None,
             "hook_evidence": "counts_and_qualified_type_only"}

    def apply():
        with timer.span("solution_apply"):
            result = original()
        timer.setup_components(candidate)
        from transformers import Trainer
        Trainer.train = capture_training(Trainer.train, state)
        # Capture metadata at the existing setup boundary, no tensor reads or
        # GPU synchronization. Exact storage addresses differ between runs;
        # record alias group memberships instead of comparing addresses.
        from soup_cli.trainer import sft
        setup = sft.SFTTrainerWrapper._setup_transformers
        def capture(self, cfg, tcfg):
            value = setup(self, cfg, tcfg)
            groups = {}
            tensors = []
            for kind, iterator in (("parameter", self.model.named_parameters(remove_duplicate=False)),
                                   ("buffer", self.model.named_buffers(remove_duplicate=False))):
                for name, tensor in iterator:
                    key = (str(tensor.device), tensor.untyped_storage().data_ptr())
                    alias = groups.setdefault(key, len(groups))
                    tensors.append({"name": name, "kind": kind, "device": str(tensor.device),
                                    "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                                    "class": type(tensor).__module__+"."+type(tensor).__qualname__,
                                    "requires_grad": tensor.requires_grad, "alias_group": alias})
                    qs = getattr(tensor, "quant_state", None)
                    if qs is not None:
                        tensors[-1]["quant_state"] = {
                            key: str(getattr(qs, key, None))
                            for key in ("quant_type", "blocksize", "dtype", "shape", "nested")}
            state["tensors"] = tensors
            state["hf_device_map"] = getattr(self.model, "hf_device_map", None)
            quant = getattr(self.model.config, "quantization_config", None)
            if hasattr(quant, "to_dict"):
                quant = quant.to_dict()
            state["quantization_config"] = quant
            state["hooks"] = [{"name": name,
                               "accelerate_hook": type(getattr(mod, "_hf_hook", None)).__module__ + "." + type(getattr(mod, "_hf_hook", None)).__qualname__,
                               "pre": len(mod._forward_pre_hooks), "post": len(mod._forward_hooks),
                               "backward": len(mod._backward_hooks),
                               "backward_pre": len(mod._backward_pre_hooks)}
                              for name, mod in self.model.named_modules()]
            state["setup_completed"] = True
            return value
        sft.SFTTrainerWrapper._setup_transformers = capture
        return result

    solution.apply = apply
    def report():
        state["components_inclusive"] = timer.records
        print("STARTUP_SETUP_DIAGNOSTIC " + json.dumps(state, sort_keys=True), flush=True)
    atexit.register(report)
    sys.argv = [str(repo / "harness/_train.py"), *harness_args]
    try:
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        assert state["training_completed"] and state["training_calls"] == 1
        assert state["completed_steps"] == 64
        state["exit_status"] = 0
        state["full_run_completed"] = True
    except BaseException:
        state["exit_status"] = 1
        raise


if __name__ == "__main__":
    main()
