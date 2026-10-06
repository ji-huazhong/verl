# Copyright 2026 Individual Contributor
# SPDX-License-Identifier: Apache-2.0
"""Deduplicate the official DAPO/AIME Parquets without changing their prompts."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REVISIONS = {
    "DAPO-Math-17k": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "AIME-2024": "aa49075e24ad594b79fdf0bdcefa735c2181be67",
}


SOURCE_SHA256 = {
    "DAPO-Math-17k": "534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941",
    "AIME-2024": "12154e38a716d12db5731f9a022ae69a610c4f7d0e0dcc04e902887a686877e7",
}


def prompt_key(row):
    return json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_rows(path):
    """Keep exact duplicates once; exclude all instances of conflicting labels."""
    source = pq.ParquetFile(path)
    by_prompt, counts, conflicts = {}, {}, {}
    for batch in source.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            prompt, reward = row["prompt"], row["reward_model"]
            if not prompt or any(item["role"] != "user" or not item["content"] for item in prompt):
                raise ValueError("Expected nonempty official user prompts")
            answer = reward["ground_truth"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Missing exact math answer")
            key = prompt_key(row)
            if key in by_prompt and by_prompt[key]["reward_model"] != reward:
                conflicts.setdefault(key, set()).update([by_prompt[key]["reward_model"]["ground_truth"], answer])
            if key in by_prompt and by_prompt[key]["data_source"] != row["data_source"]:
                raise ValueError("Identical prompt has conflicting reward dispatch")
            by_prompt.setdefault(key, row)
            counts[key] = counts.get(key, 0) + 1
    for key in conflicts:
        del by_prompt[key]
    return (
        by_prompt,
        dict(
            source_rows=source.metadata.num_rows,
            unique_prompts=len(by_prompt),
            duplicate_rows=source.metadata.num_rows - len(counts),
            conflicting_prompts_excluded=len(conflicts),
            conflicting_rows_excluded=sum(counts[key] for key in conflicts),
            conflicting_labels=[
                dict(prompt_sha256=hashlib.sha256(key.encode()).hexdigest(), answers=sorted(answers))
                for key, answers in conflicts.items()
            ],
            repeat_counts=sorted(set(counts.values())),
        ),
        source.schema_arrow,
    )


def prepare(train_source, val_source, output_dir):
    for name, path in [("DAPO-Math-17k", train_source), ("AIME-2024", val_source)]:
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != SOURCE_SHA256[name]:
                raise ValueError(f"Source differs from the pinned official {name} file")
    train, train_stats, train_schema = unique_rows(train_source)
    validation, val_stats, val_schema = unique_rows(val_source)
    overlap = train.keys() & validation.keys()
    for key in overlap:
        del train[key]
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, rows, schema in [("train.parquet", train, train_schema), ("aime2024.parquet", validation, val_schema)]:
        if not rows:
            raise ValueError("Empty dataset after deduplication")
        path = output_dir / name
        pq.write_table(pa.Table.from_pylist(list(rows.values()), schema=schema), path, compression="zstd")
        outputs[name] = dict(
            rows=len(rows), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    sources = {}
    for name, path in [("DAPO-Math-17k", train_source), ("AIME-2024", val_source)]:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        sources[name] = dict(
            repository=f"BytedTsinghua-SIA/{name}", revision=REVISIONS[name], sha256=digest, bytes=path.stat().st_size
        )
    report = dict(
        sources=sources,
        train=train_stats,
        validation=val_stats,
        outputs=outputs,
        identical_train_validation_prompts_removed=len(overlap),
        transformation="Deduplicate exact prompt/labels, exclude conflicting labels, remove exact validation overlap.",
        prompt_text_modified=False,
        overlap_scope="Exact message-text identity; not a proof of semantic benchmark decontamination.",
    )
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_source, args.val_source, args.output_dir)))


if __name__ == "__main__":
    main()
