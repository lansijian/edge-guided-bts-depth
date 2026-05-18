from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(r"D:\机器视觉设计与实践")
NYU_ROOT = PROJECT_ROOT / "data" / "nyuv2" / "official_splits"
DIODE_ROOT = PROJECT_ROOT / "data" / "diode"

NYU_FOCAL = 518.8579
DIODE_FOCAL = 715.0873


def build_nyu_list(split: str, out_path: Path) -> int:
    root = NYU_ROOT / split
    rows = []
    for rgb_path in sorted(root.rglob("rgb_*.jpg")):
        depth_path = rgb_path.with_name(rgb_path.name.replace("rgb_", "sync_depth_").replace(".jpg", ".png"))
        if not depth_path.exists():
            continue
        rel_rgb = rgb_path.relative_to(root).as_posix()
        rel_depth = depth_path.relative_to(root).as_posix()
        rows.append(f"{rel_rgb} {rel_depth} {NYU_FOCAL}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def build_diode_list(csv_path: Path, out_path: Path) -> int:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(f"{row['rgb']} {row['depth']} {DIODE_FOCAL} {row['mask']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    nyu_out = PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs"
    diode_out = PROJECT_ROOT / "data" / "diode" / "bts_inputs"

    counts = {
        "nyu_train": build_nyu_list("train", nyu_out / "nyu_train_official.txt"),
        "nyu_test": build_nyu_list("test", nyu_out / "nyu_test_official.txt"),
        "diode_train": build_diode_list(DIODE_ROOT / "splits" / "train.csv", diode_out / "diode_train.txt"),
        "diode_val": build_diode_list(DIODE_ROOT / "splits" / "val.csv", diode_out / "diode_val.txt"),
        "diode_test": build_diode_list(DIODE_ROOT / "splits" / "test.csv", diode_out / "diode_test.txt"),
    }
    print(counts)


if __name__ == "__main__":
    main()
