# Startup validation, round two

The copied overhead module is disabled. The intended experiment enables FAST_MAP only; DEVICE_MAP remains off. Credit: Alikhan-kai 44d1a47 and iencislennon 1fea4ce. No runtime promotion is established by this folder.

Run CPU tests with `STARTUP_TOKENIZER=/path/to/local/Qwen/tokenizer CUDA_VISIBLE_DEVICES="" python -m pytest test_overhead.py -q`. The local tests can use Soup's registered fallback template; they are not the strict native gate.

Strict parity requires datasets 4.8.5 and native revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Run `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python parity_gate.py --tokenizer /path/to/snapshot --trusted-tokenizer-manifest trusted-native-tokenizer.json --public-rows /repo/data/train.jsonl --public-eval /repo/data/eval_public.jsonl`. The static manifest was independently downloaded from the public fixed Hugging Face revision, verified against its Git blob IDs, and then compared with Kaggle's cache. It is not generated from candidate/adapter tokenizers.

`setup_probe.py` wraps an unchanged complete harness training invocation and requires all 64 steps. The task uses one epoch over 256 rows, so max_steps=-1 is accepted only for its exact batch/accumulation/world/epoch metadata. `compare_setup.py` compares actual tensor placement when the single-device model has no hf_device_map attribute; empty or non-CUDA placement fails. It compares hook counts/types only, not arbitrary hook behavior. All captured hooks were absent in the parent T4 runs.

Both parent runs completed 64 steps, with identical Rewind row IDs/counts and cumulative tokens. Training losses differed by at most .000888; this is not a bitwise full-training claim. Mapping diagnostic time was2.7216→.5519s. Ordinary command-time comparison and final qualification remain pending.
