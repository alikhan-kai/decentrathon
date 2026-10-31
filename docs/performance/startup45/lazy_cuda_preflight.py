"""FUTURE GPU-OWNER PREFLIGHT ONLY. Not run by this CPU implementation task.

python -B cuda_preflight.py --stage /absolute/enabled/stage --output result.json
Run in a dedicated diagnostic process, never as a benchmark warmup.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import types


def children(code):
    return {c.co_name: c for c in code.co_consts if isinstance(c, types.CodeType)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    import torch
    assert torch.cuda.is_available(), 'GPU owner must allocate one CUDA device'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.device_count() == 1
    import transformers
    receipt = json.loads((args.stage.parent / 'manifest.json').read_text())
    assert receipt['variant'] == 'lazy-bnb' and args.stage.name == 'candidate'
    for name, digest in receipt['candidate'].items():
        assert hashlib.sha256((args.stage / name).read_bytes()).hexdigest() == digest, name
    assert not any(n == 'bitsandbytes' or n.startswith('bitsandbytes.') for n in sys.modules)
    path = Path(importlib.metadata.distribution('bitsandbytes').locate_file('bitsandbytes/autograd/_functions.py')).resolve()
    source = path.read_bytes()
    expected = children(children(compile(source, str(path), 'exec', dont_inherit=True))['MatMul4Bit'])
    import solution
    original_compile = torch.compile
    start = time.perf_counter()
    solution.apply()
    apply_s = time.perf_counter() - start
    from solution import lazy_bnb_compile as lazy
    after_apply = lazy.stats()
    assert torch.compile is original_compile
    assert len(after_apply['functions']) == 4, after_apply
    assert all(r['calls'] == r['constructions'] == 0 for r in after_apply['functions']), after_apply
    # Do not race the existing donor precompile worker during this diagnostic.
    from solution import precompile
    worker = getattr(precompile, '_thread', None)
    if worker is not None:
        worker.join(timeout=90)
        assert not worker.is_alive(), 'Existing donor precompile worker timed out'
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    from bitsandbytes.autograd._functions import MatMul4Bit
    assert MatMul4Bit.forward.__code__ == expected['forward']
    assert MatMul4Bit.backward.__code__ == expected['backward']
    assert path.read_bytes() == source
    torch.manual_seed(7)
    weights = torch.randn(32, 64, device='cuda', dtype=torch.float16)
    packed, state = F.quantize_4bit(weights, blocksize=64, compress_statistics=True, quant_type='nf4')
    assert state.nested and state.quant_type == 'nf4' and state.blocksize == 64
    # M=2048 exercises the training dequantize+linear path, not GEMV inference.
    x = torch.randn(2048, 64, device='cuda', dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2048, 32, device='cuda', dtype=torch.float16)
    dense = F.dequantize_4bit(packed, state)
    expected_out = torch.nn.functional.linear(x.detach(), dense)
    expected_grad = grad @ dense
    actual = bnb.matmul_4bit(x, packed.t(), quant_state=state)
    assert 'MatMul4BitBackward' in type(actual.grad_fn).__name__, type(actual.grad_fn)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)
    torch.cuda.synchronize()
    after_nf4 = lazy.stats()
    assert all(r['calls'] == r['constructions'] == 0 for r in after_nf4['functions']), after_nf4
    report = {'apply_s_diagnostic_only': apply_s, 'after_apply': after_apply,
              'after_true_nf4_autograd': after_nf4,
              'compiler_restored': torch.compile is original_compile,
              'true_nf4_autograd_parity': True,
              'bnb_autograd_source_sha256': hashlib.sha256(source).hexdigest(),
              'torch': torch.__version__, 'bnb': bnb.__version__,
              'device': torch.cuda.get_device_name(0)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
