"""Other training and eval rows for a robustness run (different rows, same dataset and revision as task/prepare_data.py).

    python tools/make_robust_data.py [--out data_robust] [--train-start 1000] [--eval-start 41000]

Writes <out>/train.jsonl (256 rows from --train-start) and <out>/eval.jsonl (256 rows from --eval-start), using the
dataset, revision and writer of task/prepare_data.py (imported, that file is not changed). The public ranges are
0..255 (train) and 40000..40255 (eval), so the defaults do not overlap them.
"""

import argparse
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "data_robust"))
    parser.add_argument("--train-start", type=int, default=1000)
    parser.add_argument("--eval-start", type=int, default=41000)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("prepare_data", REPO / "task" / "prepare_data.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    from datasets import load_dataset

    dataset = load_dataset(prepare.DATASET, split="train", revision=prepare.REVISION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepare.write_jsonl(out / "train.jsonl", (dataset[i] for i in range(args.train_start, args.train_start + args.rows)))
    prepare.write_jsonl(out / "eval.jsonl", (dataset[i] for i in range(args.eval_start, args.eval_start + args.rows)))
    print(f"wrote {args.rows} train rows from {args.train_start} and {args.rows} eval rows from {args.eval_start} to {out}")


if __name__ == "__main__":
    main()
