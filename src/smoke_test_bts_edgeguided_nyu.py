import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BTS_ROOT = PROJECT_ROOT / "refs" / "bts" / "pytorch"

sys.path.insert(0, str(BTS_ROOT))
os.chdir(BTS_ROOT)

from bts import BtsModel, silog_loss  # noqa: E402


class DummyArgs:
    encoder = "densenet121_bts"
    bts_size = 512
    max_depth = 10
    dataset = "nyu"
    use_edge_guidance = True
    edge_loss_weight = 0.1


def compute_depth_edge_target(depth, mask, max_depth):
    depth_norm = torch.clamp(depth / max_depth, min=0.0, max=1.0)

    grad_x = torch.abs(depth_norm[:, :, :, 1:] - depth_norm[:, :, :, :-1])
    grad_y = torch.abs(depth_norm[:, :, 1:, :] - depth_norm[:, :, :-1, :])

    mask_x = torch.logical_and(mask[:, :, :, 1:], mask[:, :, :, :-1])
    mask_y = torch.logical_and(mask[:, :, 1:, :], mask[:, :, :-1, :])

    edge_x = torch.zeros_like(depth_norm)
    edge_y = torch.zeros_like(depth_norm)
    edge_x[:, :, :, 1:] = grad_x * mask_x.float()
    edge_y[:, :, 1:, :] = grad_y * mask_y.float()

    edge_target = torch.maximum(edge_x, edge_y)
    edge_target = (edge_target > 0.03).float()
    return edge_target


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    args = DummyArgs()
    model = BtsModel(args).cuda()
    model.train()

    image = torch.randn(2, 3, 416, 544, device="cuda")
    focal = torch.full((2,), 518.8579, device="cuda")
    depth_gt = torch.rand(2, 1, 416, 544, device="cuda") * 9.5 + 0.2
    mask = torch.logical_and(depth_gt > 0.1, depth_gt < args.max_depth)

    outputs = model(image, focal)
    assert len(outputs) == 6
    _, _, _, _, depth_est, edge_logits = outputs

    depth_loss = silog_loss(variance_focus=0.85).forward(depth_est, depth_gt, mask)
    edge_target = compute_depth_edge_target(depth_gt, mask, args.max_depth)
    edge_loss = torch.nn.BCEWithLogitsLoss()(edge_logits, edge_target)
    total_loss = depth_loss + args.edge_loss_weight * edge_loss
    total_loss.backward()

    print(f"depth_shape={tuple(depth_est.shape)}")
    print(f"edge_shape={tuple(edge_logits.shape)}")
    print(f"depth_loss={depth_loss.item():.6f}")
    print(f"edge_loss={edge_loss.item():.6f}")
    print(f"total_loss={total_loss.item():.6f}")
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
