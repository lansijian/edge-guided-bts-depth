from __future__ import annotations

import csv
import random
from pathlib import Path


ROOT = Path(r"D:\机器视觉设计与实践\data\diode")
VAL_ROOT = ROOT / "val"
SPLIT_DIR = ROOT / "splits"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
COURSE_SUBSET_SIZE = 256


def collect_records(domain: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for rgb_path in sorted((VAL_ROOT / domain).rglob("*.png")):
        stem = rgb_path.stem
        depth_path = rgb_path.with_name(f"{stem}_depth.npy")
        mask_path = rgb_path.with_name(f"{stem}_depth_mask.npy")
        if not depth_path.exists() or not mask_path.exists():
            continue
        records.append(
            {
                "source_split": "val",
                "domain": domain,
                "rgb": str(rgb_path.relative_to(ROOT)).replace("\\", "/"),
                "depth": str(depth_path.relative_to(ROOT)).replace("\\", "/"),
                "mask": str(mask_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return records


def split_records(records: list[dict[str, str]], rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test


def write_csv(path: Path, records: list[dict[str, str]], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in records:
        row_copy = dict(row)
        row_copy["split"] = split_name
        rows.append(row_copy)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_split", "domain", "rgb", "depth", "mask"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    indoors = collect_records("indoors")
    outdoor = collect_records("outdoor")

    indoor_train, indoor_val, indoor_test = split_records(indoors, rng)
    outdoor_train, outdoor_val, outdoor_test = split_records(outdoor, rng)

    train = indoor_train + outdoor_train
    val = indoor_val + outdoor_val
    test = indoor_test + outdoor_test
    all_records = train + val + test

    write_csv(SPLIT_DIR / "train.csv", train, "train")
    write_csv(SPLIT_DIR / "val.csv", val, "val")
    write_csv(SPLIT_DIR / "test.csv", test, "test")
    write_csv(SPLIT_DIR / "all.csv", all_records, "mixed")

    course_subset = indoor_train[: min(COURSE_SUBSET_SIZE, len(indoor_train))]
    write_csv(SPLIT_DIR / "course_subset_indoors_256_train.csv", course_subset, "train")

    print(
        {
            "indoors_total": len(indoors),
            "outdoor_total": len(outdoor),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "all": len(all_records),
            "course_subset_indoors_256_train": len(course_subset),
        }
    )


if __name__ == "__main__":
    main()
