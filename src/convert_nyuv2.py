from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


MAT_PATH = Path(r"D:\机器视觉设计与实践\data\nyuv2\nyu_depth_v2_labeled.mat")
ROOT = MAT_PATH.parent
OUT_ROOT = ROOT / "processed"
RGB_DIR = OUT_ROOT / "rgb"
DEPTH_DIR = OUT_ROOT / "depth"
LABEL_DIR = OUT_ROOT / "label"
SPLIT_DIR = ROOT / "splits"


def ensure_dirs() -> None:
    for path in [RGB_DIR, DEPTH_DIR, LABEL_DIR, SPLIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def convert_image(arr: np.ndarray) -> np.ndarray:
    # Source shape: (3, 640, 480) -> target: (480, 640, 3)
    return np.transpose(arr, (2, 1, 0))


def convert_map(arr: np.ndarray) -> np.ndarray:
    # Source shape: (640, 480) -> target: (480, 640)
    return np.transpose(arr, (1, 0))


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "rgb", "depth", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, str]] = []

    with h5py.File(MAT_PATH, "r") as f:
        images = f["images"]
        depths = f["depths"]
        labels = f["labels"]
        total = images.shape[0]

        for idx in range(total):
            file_id = f"{idx:04d}"
            rgb = convert_image(images[idx]).astype(np.uint8)
            depth = convert_map(depths[idx]).astype(np.float32)
            label = convert_map(labels[idx]).astype(np.uint16)

            rgb_path = RGB_DIR / f"{file_id}.png"
            depth_path = DEPTH_DIR / f"{file_id}.npy"
            label_path = LABEL_DIR / f"{file_id}.npy"

            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            np.save(label_path, label)

            rows.append(
                {
                    "index": str(idx),
                    "rgb": str(rgb_path.relative_to(ROOT)).replace("\\", "/"),
                    "depth": str(depth_path.relative_to(ROOT)).replace("\\", "/"),
                    "label": str(label_path.relative_to(ROOT)).replace("\\", "/"),
                }
            )

            if (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"converted {idx + 1}/{total}")

    write_split_csv(SPLIT_DIR / "all_samples.csv", rows)

    n = len(rows)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    write_split_csv(SPLIT_DIR / "train.csv", rows[:train_end])
    write_split_csv(SPLIT_DIR / "val.csv", rows[train_end:val_end])
    write_split_csv(SPLIT_DIR / "test.csv", rows[val_end:])

    print(
        {
            "total": n,
            "train": train_end,
            "val": val_end - train_end,
            "test": n - val_end,
        }
    )


if __name__ == "__main__":
    main()
