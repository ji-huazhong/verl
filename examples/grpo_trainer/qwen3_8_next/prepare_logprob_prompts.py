# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Save reproducible GSM8K token sequences, including a sparse-attention case."""

import argparse
import json
from pathlib import Path


def main():
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--long-length", type=int, default=2304)
    args = parser.parse_args()
    if args.samples < 1 or args.long_length <= 2051:
        parser.error("Use at least one GSM8K sample and a long case exceeding the 2048+3 QSA dense boundary")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    rows = parquet.read_table(args.data, columns=["prompt", "extra_info"]).to_pylist()
    prompts, history = [], []
    for index, row in enumerate(rows):
        messages = list(row["prompt"])
        answer = (row.get("extra_info") or {}).get("answer")
        if answer:
            messages.append({"role": "assistant", "content": answer})
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=not bool(answer), return_dict=False
        )
        if len(tokens) > 3072:
            continue
        prompts.append({"id": f"gsm8k-{index}", "input_ids": tokens})
        history.extend(messages)
        if len(prompts) >= args.samples:
            break
    if not prompts:
        raise ValueError("No GSM8K examples fit the validation context")
    tokens = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=False, return_dict=False)
    # Repeating tokenized history deliberately retains EOS boundaries; PLE
    # must reset its n-gram context there on both backends.
    repeats = (args.long_length + len(tokens) - 1) // len(tokens)
    prompts.append({"id": "gsm8k-long-history", "input_ids": (tokens * repeats)[: args.long_length]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    partial.write_text(json.dumps(prompts, ensure_ascii=False) + "\n")
    partial.replace(args.output)
    print(json.dumps({"prompts": len(prompts), "lengths": [len(p["input_ids"]) for p in prompts]}))


if __name__ == "__main__":
    main()
