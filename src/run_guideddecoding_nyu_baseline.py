import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_ROOT = PROJECT_ROOT / "refs" / "GuidedDecoding"


def main():
    env = os.environ.copy()
    env["TORCH_HOME"] = str(PROJECT_ROOT / "outputs" / "cache" / "torch")
    env["MPLBACKEND"] = "Agg"

    command = [
        sys.executable,
        "main.py",
        "--train",
        "--dataset", "nyu",
        "--resolution", "half",
        "--model", "GuideDepth-S",
        "--data_path", str(PROJECT_ROOT / "data" / "nyuv2" / "official_splits" / "train"),
        "--test_path", str(PROJECT_ROOT / "data" / "nyuv2" / "official_splits" / "test"),
        "--train_list", str(PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs" / "nyu_train_official.txt"),
        "--val_list", str(PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs" / "nyu_test_official.txt"),
        "--num_workers", "0",
        "--batch_size", "4",
        "--num_epochs", "20",
        "--save_checkpoint", str(PROJECT_ROOT / "outputs" / "logs" / "guideddecoding" / "nyu" / "checkpoints"),
        "--save_results", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "nyu"),
    ]

    raise SystemExit(subprocess.call(command, cwd=GUIDE_ROOT, env=env))


if __name__ == "__main__":
    main()
