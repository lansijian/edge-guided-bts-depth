import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(r"D:\机器视觉设计与实践")
GUIDE_ROOT = PROJECT_ROOT / "refs" / "GuidedDecoding"


def main():
    env = os.environ.copy()
    env["TORCH_HOME"] = str(PROJECT_ROOT / "outputs" / "cache" / "torch")
    env["MPLBACKEND"] = "Agg"

    command = [
        sys.executable,
        "main.py",
        "--eval",
        "--dataset", "nyu",
        "--resolution", "half",
        "--model", "GuideDepth-S",
        "--weights_path", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "nyu" / "best_model.pth"),
        "--test_path", str(PROJECT_ROOT / "data" / "nyuv2" / "official_splits" / "test"),
        "--test_list", str(PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs" / "nyu_test_official.txt"),
        "--num_workers", "0",
        "--save_results", str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "nyu_eval"),
    ]

    raise SystemExit(subprocess.call(command, cwd=GUIDE_ROOT, env=env))


if __name__ == "__main__":
    main()
