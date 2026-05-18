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
        "--dataset", "diode",
        "--resolution", "half",
        "--model", "GuideDepth-S",
        "--data_path", str(PROJECT_ROOT / "data" / "diode"),
        "--test_path", str(PROJECT_ROOT / "data" / "diode"),
        "--train_list", str(PROJECT_ROOT / "data" / "diode" / "splits" / "train.csv"),
        "--val_list", str(PROJECT_ROOT / "data" / "diode" / "splits" / "val.csv"),
        "--num_workers", "0",
        "--batch_size", "4",
        "--num_epochs", "20",
        "--save_checkpoint", str(PROJECT_ROOT / "outputs" / "logs" / "guideddecoding" / "diode" / "checkpoints"),
        "--save_results", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "diode"),
    ]

    raise SystemExit(subprocess.call(command, cwd=GUIDE_ROOT, env=env))


if __name__ == "__main__":
    main()
