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
        "--mode", "train",
        "--model_name", "bts_diode_edgeguided_smoke",
        "--encoder", "densenet121_bts",
        "--dataset", "diode",
        "--data_path", str(PROJECT_ROOT / "data" / "diode"),
        "--gt_path", str(PROJECT_ROOT / "data" / "diode"),
        "--filenames_file", str(PROJECT_ROOT / "data" / "diode" / "bts_inputs" / "diode_train.txt"),
        "--batch_size", "2",
        "--num_epochs", "1",
        "--learning_rate", "1e-4",
        "--weight_decay", "1e-2",
        "--adam_eps", "1e-3",
        "--num_threads", "0",
        "--input_height", "416",
        "--input_width", "544",
        "--max_depth", "10",
        "--log_directory", str(PROJECT_ROOT / "outputs" / "logs" / "bts" / "diode"),
        "--log_freq", "50",
        "--save_freq", "200",
        "--do_online_eval",
        "--eval_freq", "200",
        "--data_path_eval", str(PROJECT_ROOT / "data" / "diode"),
        "--gt_path_eval", str(PROJECT_ROOT / "data" / "diode"),
        "--filenames_file_eval", str(PROJECT_ROOT / "data" / "diode" / "bts_inputs" / "diode_val.txt"),
        "--min_depth_eval", "1e-3",
        "--max_depth_eval", "10",
        "--eval_summary_directory", str(PROJECT_ROOT / "outputs" / "logs" / "bts" / "diode" / "eval"),
        "--use_edge_guidance",
        "--edge_loss_weight", "0.05",
    ]
    raise SystemExit(subprocess.call(command, cwd=BTS_ROOT, env=env))


if __name__ == "__main__":
    main()
