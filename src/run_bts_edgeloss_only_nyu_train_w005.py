import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BTS_ROOT = PROJECT_ROOT / "refs" / "bts" / "pytorch"


def main():
    env = os.environ.copy()
    command = [
        sys.executable,
        str(BTS_ROOT / "bts_main.py"),
        str(PROJECT_ROOT / "configs" / "bts" / "arguments_train_nyu_edgeloss_only_w005.txt"),
    ]
    raise SystemExit(subprocess.call(command, cwd=BTS_ROOT, env=env))


if __name__ == "__main__":
    main()
