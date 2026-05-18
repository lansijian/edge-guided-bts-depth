import os
import sys
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(r"D:\机器视觉设计与实践")
GUIDE_ROOT = PROJECT_ROOT / "refs" / "GuidedDecoding"

os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "outputs" / "cache" / "torch"))
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(GUIDE_ROOT))
os.chdir(GUIDE_ROOT)

from training import Trainer  # noqa: E402


def main():
    args = Namespace(
        save_checkpoint=str(PROJECT_ROOT / "outputs" / "logs" / "guideddecoding" / "nyu_smoke" / "checkpoints"),
        save_results=str(PROJECT_ROOT / "outputs" / "results" / "guideddecoding" / "nyu_smoke"),
        num_epochs=1,
        dataset="nyu",
        model="GuideDepth-S",
        weights_path=None,
        data_path=str(PROJECT_ROOT / "data" / "nyuv2" / "official_splits" / "train"),
        test_path=str(PROJECT_ROOT / "data" / "nyuv2" / "official_splits" / "test"),
        train_list=str(PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs" / "nyu_train_official.txt"),
        val_list=str(PROJECT_ROOT / "data" / "nyuv2" / "bts_inputs" / "nyu_test_official.txt"),
        test_list="",
        eval_mode="alhashim",
        batch_size=2,
        resolution="half",
        num_workers=0,
        learning_rate=1e-4,
        scheduler_step_size=15,
        load_checkpoint="",
    )

    trainer = Trainer(args)
    print(f"train_len={len(trainer.train_loader.dataset)}")
    print(f"val_len={len(trainer.val_loader.dataset)}")

    batch = next(iter(trainer.train_loader))
    image, depth = trainer.unpack_and_move(batch)
    prediction = trainer.model(image)
    loss = trainer.loss_func(prediction, depth)
    loss.backward()
    val_batch = next(iter(trainer.val_loader))
    val_image, val_depth = trainer.unpack_and_move(val_batch)
    print(f"image_shape={tuple(image.shape)}")
    print(f"depth_shape={tuple(depth.shape)}")
    print(f"pred_shape={tuple(prediction.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"val_image_shape={tuple(val_image.shape)}")
    print(f"val_depth_shape={tuple(val_depth.shape)}")
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
