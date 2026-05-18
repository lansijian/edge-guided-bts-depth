import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BTS_ROOT = PROJECT_ROOT / "refs" / "bts" / "pytorch"


def main():
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        str(BTS_ROOT / "bts_test.py"),
        str(PROJECT_ROOT / "configs" / "bts" / "arguments_test_diode_edgeguided_w002.txt"),
    ]
    raise SystemExit(subprocess.call(command, cwd=BTS_ROOT, env=env))


if __name__ == "__main__":
    main()
