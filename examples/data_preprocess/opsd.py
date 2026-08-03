#!/usr/bin/env python3
"""Preprocess OpenThoughts Math OPSD into a verl-compatible Parquet file."""

import argparse
import os

import datasets


DATASET_ID = "siyanzhao/Openthoughts_math_30k_opsd"
DATA_SOURCE = "math"


def make_map_fn(split):
    def process_fn(example, idx):
        problem = example["problem"].strip()
        solution = example["solution"].strip()

        return {
            # "math" selects verl's built-in math reward scorer. The original
            # Hugging Face dataset name is retained in extra_info below.
            "data_source": DATA_SOURCE,
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"Problem: {problem}\n\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                }
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": (example["Answer"] or "").strip(),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "problem": problem,
                "solution": solution,
                "dataset_id": DATASET_ID,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="~/data/openthoughts_math_opsd",
        help="The save directory for the preprocessed dataset.",
    )
    args = parser.parse_args()

    dataset = datasets.load_dataset(DATASET_ID)
    source_dataset = dataset["train"]
    train_dataset = source_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=source_dataset.column_names,
    )

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

