"""Manually download pinned GSM8K and write disjoint fixed local subsets."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lockstep_rl.data import validate_splits
from lockstep_rl.rewards import extract_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("Dataset download requires manual --allow-download")
    if min(args.train_count, args.validation_count) < 1:
        parser.error("subset counts must be positive")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = HfApi().dataset_info("openai/gsm8k", revision=args.revision).sha
    dataset = load_dataset("openai/gsm8k", "main", revision=revision)
    subsets = {}
    for output_split, source_split, count in (("train", "train", args.train_count),
                                               ("validation", "test", args.validation_count)):
        if count > len(dataset[source_split]):
            raise ValueError("requested subset exceeds source split")
        indices = random.Random(args.seed).sample(range(len(dataset[source_split])), count)
        rows = []
        for index in indices:
            row = dataset[source_split][index]
            answer = extract_answer(row["answer"])
            if answer is None:
                raise ValueError("GSM8K gold answer did not match the documented numeric parser")
            rows.append({"id": f"gsm8k-{source_split}-{index}",
                         "prompt": row["question"] + "\nEnd with <answer>number</answer>.",
                         "answer": str(answer)})
        subsets[output_split] = rows
    validate_splits(subsets["train"], subsets["validation"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in subsets.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"dataset": "openai/gsm8k",
         "revision": revision, "seed": args.seed, "validation_source": "test",
         "train_count": args.train_count, "validation_count": args.validation_count}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
