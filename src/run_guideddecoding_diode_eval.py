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
        "--eval",
        "--dataset", "diode",
        "--resolution", "half",
        "--model", "GuideDepth-S",
        "--weights_path", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "diode" / "best_model.pth"),
        "--test_path", str(PROJECT_ROOT / "data" / "diode"),
        "--test_list", str(PROJECT_ROOT / "data" / "diode" / "splits" / "test.csv"),
        "--num_workers", "0",
        "--save_results", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "diode_eval"),
    ]

    raise SystemExit(subprocess.call(command, cwd=GUIDE_ROOT, env=env))


if __name__ == "__main__":
    main()
