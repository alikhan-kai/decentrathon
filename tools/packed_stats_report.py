"""R/P per step of the packed forward, read from train.log.   python3 tools/packed_stats_report.py runs/solution_1/train.log

solution/packed.py prints one line "PACKED_STATS {json}" when the training process exits. R = real tokens of a step,
P = padded tokens (batch * longest row); R/P is the share of the padded work that was real. Standard library only.
"""

import json
import sys

PREFIX = "PACKED_STATS "


def read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return [json.loads(line[line.index(PREFIX) + len(PREFIX):]) for line in handle if PREFIX in line]


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    for path in argv[1:]:
        records = read(path)
        if not records:
            print(f"{path}: no PACKED_STATS line (packed.py not active, or the process was killed)")
            continue
        for i, rec in enumerate(records):
            real, padded = rec["n_real"], rec["n_padded"]
            print(f"{path} [process {i}]: packed_steps={rec['packed_steps']} fallbacks={rec['fallbacks']}")
            for step, (r, p) in enumerate(zip(real, padded), 1):
                print(f"  step {step:4d}  R={r:6d}  P={p:6d}  R/P={r / p:.3f}")
            if padded:
                print(f"  total R={sum(real)} P={sum(padded)} R/P={sum(real) / sum(padded):.3f} "
                      f"min={min(r / p for r, p in zip(real, padded)):.3f} max={max(r / p for r, p in zip(real, padded)):.3f}")


if __name__ == "__main__":
    main(sys.argv)
