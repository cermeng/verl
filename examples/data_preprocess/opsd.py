# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the public OPSD math data for verl.

The output deliberately stores two chat prompts per row:

* ``prompt`` is visible to the student and contains only the problem.
* ``opsd_teacher_prompt`` additionally contains the verified reference
  solution.  The teacher scores the student's sampled response under this
  privileged context; it does not generate a second response.

No tokenizer is used here, so the parquet files remain model-independent.
Chat templates are applied by the rollout workers at runtime.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


DATA_SOURCE = "siyanzhao/Openthoughts_math_30k_opsd"
TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def make_map_fn(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        problem = example["problem"]
        solution = example["solution"]
        boxed_answer = last_boxed_only_string(solution)
        ground_truth = remove_boxed(boxed_answer) if boxed_answer is not None else solution

        # Match the reference repository's non-reason-first collator prompts.
        student_prompt = (
            f"Problem: {problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        teacher_prompt = (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
            f"{TRANSITION_PROMPT}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

        return {
            # The generic PPO data path still invokes a reward manager even
            # though OPSD does not consume task rewards. Route through verl's
            # built-in math scorer so this bookkeeping step remains defined.
            "data_source": "lighteval/MATH",
            "prompt": [{"role": "user", "content": student_prompt}],
            "opsd_teacher_prompt": [{"role": "user", "content": teacher_prompt}],
            "ability": "math",
            # The current PPO data path still materializes rewards/advantages even
            # when OPSD sets use_task_rewards=false.  The value is not part of the
            # OPSD loss, but retaining it keeps the standard math reward interface.
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "source_dataset": DATA_SOURCE,
                "problem": problem,
            },
        }

    return process_fn


def format_split(dataset: datasets.Dataset, split: str) -> datasets.Dataset:
    return dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting OPSD {split} split",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local dataset path. If omitted, load the public Hugging Face dataset.",
    )
    parser.add_argument("--local_save_dir", default="~/data/opsd")
    parser.add_argument("--validation_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.local_dataset_path or DATA_SOURCE)
    if isinstance(dataset, datasets.DatasetDict):
        train_dataset = dataset["train"]
        if "validation" in dataset:
            validation_dataset = dataset["validation"]
        elif "test" in dataset:
            validation_dataset = dataset["test"]
        else:
            validation_dataset = None
    else:
        train_dataset = dataset
        validation_dataset = None

    if validation_dataset is None:
        if not 0 < args.validation_size < len(train_dataset):
            raise ValueError(
                f"validation_size must be in (0, {len(train_dataset)}), got {args.validation_size}."
            )
        split = train_dataset.train_test_split(test_size=args.validation_size, seed=args.seed, shuffle=True)
        train_dataset, validation_dataset = split["train"], split["test"]

    train_dataset = format_split(train_dataset, "train")
    validation_dataset = format_split(validation_dataset, "validation")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    validation_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
