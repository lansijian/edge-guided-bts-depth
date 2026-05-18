from __future__ import annotations

import csv
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tensorboardX.proto import event_pb2


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSET_DIR = REPORT_DIR / "assets"
FIG_DIR = ASSET_DIR / "figures"
TABLE_DIR = ASSET_DIR / "tables"
PROMPT_DIR = ASSET_DIR / "ai_prompts"


@dataclass
class ScalarPoint:
    step: int
    value: float


def ensure_dirs() -> None:
    for path in [FIG_DIR, TABLE_DIR, PROMPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_event_scalars(event_path: Path) -> dict[str, list[ScalarPoint]]:
    scalars: dict[str, list[ScalarPoint]] = {}
    with event_path.open("rb") as f:
        while True:
            header = f.read(8)
            if not header or len(header) < 8:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # length crc
            data = f.read(length)
            f.read(4)  # data crc
            event = event_pb2.Event()
            event.ParseFromString(data)
            if not event.summary.value:
                continue
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    scalars.setdefault(value.tag, []).append(
                        ScalarPoint(step=event.step, value=float(value.simple_value))
                    )
    return scalars


def latest_event_file(folder: Path) -> Path:
    files = sorted(folder.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No event file found in {folder}")
    return files[-1]


def write_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def plot_curve_comparison(
    dataset: str,
    baseline_train: Path,
    baseline_eval: Path,
    improved_train: Path,
    improved_eval: Path,
    output_path: Path,
) -> None:
    base_train_scalars = read_event_scalars(latest_event_file(baseline_train))
    base_eval_scalars = read_event_scalars(latest_event_file(baseline_eval))
    imp_train_scalars = read_event_scalars(latest_event_file(improved_train))
    imp_eval_scalars = read_event_scalars(latest_event_file(improved_eval))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    def xy(points: list[ScalarPoint]) -> tuple[list[int], list[float]]:
        return [p.step for p in points], [p.value for p in points]

    x, y = xy(base_train_scalars["silog_loss"])
    axes[0].plot(x, y, label="Baseline silog_loss", linewidth=2)
    x, y = xy(imp_train_scalars["silog_loss"])
    axes[0].plot(x, y, label="Edge-Guided silog_loss", linewidth=2)
    if "edge_loss" in imp_train_scalars:
        x, y = xy(imp_train_scalars["edge_loss"])
        axes[0].plot(x, y, label="Edge-Guided edge_loss", linewidth=1.5, linestyle="--")
    axes[0].set_title(f"{dataset} Training Curves")
    axes[0].set_xlabel("Global step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    x, y = xy(base_eval_scalars["abs_rel"])
    axes[1].plot(x, y, label="Baseline abs_rel", linewidth=2)
    x, y = xy(imp_eval_scalars["abs_rel"])
    axes[1].plot(x, y, label="Edge-Guided abs_rel", linewidth=2)
    x, y = xy(base_eval_scalars["silog"])
    axes[1].plot(x, y, label="Baseline silog", linewidth=1.5, linestyle=":")
    x, y = xy(imp_eval_scalars["silog"])
    axes[1].plot(x, y, label="Edge-Guided silog", linewidth=1.5, linestyle=":")
    axes[1].set_title(f"{dataset} Online Eval Curves")
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("Metric value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def normalize_nyu_name(name: str) -> str:
    return name.replace("__", "_")


def normalize_identity(name: str) -> str:
    return name


def strip_scale_suffix(name: str) -> str:
    return re.sub(r"_(1x1|2x2|4x4|8x8)(?=\.[^.]+$)", "", name)


def pick_common_prediction_files(
    folder_a: Path,
    folder_b: Path,
    normalizer,
    limit: int = 3,
) -> list[str]:
    names_a = {
        normalizer(strip_scale_suffix(p.name)): p.name
        for p in folder_a.iterdir()
        if p.is_file() and re.search(r"\.(png|jpg|jpeg)$", p.name, re.IGNORECASE)
    }
    names_b = {
        normalizer(strip_scale_suffix(p.name)): p.name
        for p in folder_b.iterdir()
        if p.is_file()
        and re.search(r"\.(png|jpg|jpeg)$", p.name, re.IGNORECASE)
        and not re.search(r"_(1x1|2x2|4x4|8x8)\.", p.name)
    }
    common = sorted(set(names_a) & set(names_b))
    return common[:limit]


def load_image(path: Path, target_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize(target_size)


def find_rgb_path(rgb_dir: Path, common_name: str, normalizer) -> Path:
    stem = Path(common_name).stem
    candidates = [
        rgb_dir / f"{stem}.jpg",
        rgb_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in rgb_dir.iterdir():
        if candidate.is_file() and normalizer(candidate.name) == common_name:
            return candidate
    raise FileNotFoundError(f"No RGB image for {common_name} in {rgb_dir}")


def make_visual_comparison(
    title: str,
    rgb_dir: Path,
    baseline_cmap_dir: Path,
    improved_cmap_dir: Path,
    output_path: Path,
    normalizer,
    limit: int = 3,
) -> None:
    samples = pick_common_prediction_files(baseline_cmap_dir, improved_cmap_dir, normalizer=normalizer, limit=limit)
    panel_w, panel_h = 320, 240
    margin = 18
    header_h = 64
    row_h = panel_h + 70
    width = margin * 4 + panel_w * 3
    height = header_h + len(samples) * row_h + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill="black")
    headers = ["RGB", "Baseline", "Edge-Guided"]
    for j, header in enumerate(headers):
        draw.text((margin + j * (panel_w + margin), 42), header, fill="black")

    for i, normalized in enumerate(samples):
        y = header_h + i * row_h
        baseline_name = next(
            p.name
            for p in baseline_cmap_dir.iterdir()
            if p.is_file()
            and normalizer(strip_scale_suffix(p.name)) == normalized
            and not re.search(r"_(1x1|2x2|4x4|8x8)\.", p.name)
        )
        improved_name = next(
            p.name
            for p in improved_cmap_dir.iterdir()
            if p.is_file()
            and normalizer(strip_scale_suffix(p.name)) == normalized
            and not re.search(r"_(1x1|2x2|4x4|8x8)\.", p.name)
        )
        rgb = load_image(find_rgb_path(rgb_dir, normalized, normalizer), (panel_w, panel_h))
        baseline = load_image(baseline_cmap_dir / baseline_name, (panel_w, panel_h))
        improved = load_image(improved_cmap_dir / improved_name, (panel_w, panel_h))
        row_images = [rgb, baseline, improved]
        for j, image in enumerate(row_images):
            x = margin + j * (panel_w + margin)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + panel_w, y + panel_h), outline="black", width=1)
        draw.text((margin, y + panel_h + 10), f"Sample {i + 1}: {Path(normalized).stem}", fill="black")

    canvas.save(output_path)


def make_problem_analysis_figure(
    rgb_dir: Path,
    baseline_cmap_dir: Path,
    improved_cmap_dir: Path,
    output_path: Path,
) -> None:
    sample = pick_common_prediction_files(
        baseline_cmap_dir,
        improved_cmap_dir,
        normalizer=normalize_nyu_name,
        limit=1,
    )[0]
    baseline_name = next(
        p.name
        for p in baseline_cmap_dir.iterdir()
        if p.is_file()
        and normalize_nyu_name(strip_scale_suffix(p.name)) == sample
        and not re.search(r"_(1x1|2x2|4x4|8x8)\.", p.name)
    )
    improved_name = next(
        p.name
        for p in improved_cmap_dir.iterdir()
        if p.is_file()
        and normalize_nyu_name(strip_scale_suffix(p.name)) == sample
        and not re.search(r"_(1x1|2x2|4x4|8x8)\.", p.name)
    )
    rgb = Image.open(find_rgb_path(rgb_dir, sample, normalize_nyu_name)).convert("RGB")
    baseline = Image.open(baseline_cmap_dir / baseline_name).convert("RGB")
    improved = Image.open(improved_cmap_dir / improved_name).convert("RGB")

    crop_box = (
        rgb.width // 4,
        rgb.height // 4,
        rgb.width // 4 + rgb.width // 3,
        rgb.height // 4 + rgb.height // 3,
    )
    rgb_marked = rgb.copy()
    draw = ImageDraw.Draw(rgb_marked)
    draw.rectangle(crop_box, outline="red", width=4)

    crop_rgb = rgb.crop(crop_box).resize((300, 220))
    crop_base = baseline.crop(crop_box).resize((300, 220))
    crop_improved = improved.crop(crop_box).resize((300, 220))

    panels = [
        rgb_marked.resize((360, 270)),
        baseline.resize((360, 270)),
        improved.resize((360, 270)),
        crop_rgb,
        crop_base,
        crop_improved,
    ]
    width = 3 * 380 + 30
    height = 2 * 300 + 50
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    titles = [
        "Input RGB with boundary region",
        "Baseline prediction",
        "Edge-Guided prediction",
        "Boundary crop (RGB)",
        "Boundary crop (Baseline)",
        "Boundary crop (Edge-Guided)",
    ]
    for idx, (panel, title) in enumerate(zip(panels, titles)):
        row, col = divmod(idx, 3)
        x = 10 + col * 380
        y = 10 + row * 300
        canvas.paste(panel, (x, y + 20))
        draw.text((x, y), title, fill="black")
        draw.rectangle((x, y + 20, x + panel.width, y + 20 + panel.height), outline="black", width=1)

    canvas.save(output_path)


def plot_ablation_bars(
    dataset: str,
    groups: list[str],
    abs_rel: list[float],
    rms: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    for ax, values, name in zip(axes, [abs_rel, rms], ["abs_rel", "rms"]):
        ax.bar(groups, values, color=colors[: len(groups)])
        ax.set_title(f"{dataset} {name}")
        ax.set_ylabel(name)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_ablation(output_path: Path) -> None:
    groups = ["baseline", "branch", "loss", "full-0.02", "full-0.05"]
    nyu_abs_rel = [0.168, 0.171, 0.163, 0.166, 0.166]
    diode_abs_rel = [0.542, 0.604, 0.540, 0.516, 0.558]
    nyu_rms = [0.547, 0.539, 0.543, 0.545, 0.537]
    diode_rms = [1.773, 1.980, 1.754, 1.741, 1.832]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    x = list(range(len(groups)))
    for ax, nyu_vals, diode_vals, metric in [
        (axes[0], nyu_abs_rel, diode_abs_rel, "abs_rel"),
        (axes[1], nyu_rms, diode_rms, "rms"),
    ]:
        ax.plot(x, nyu_vals, marker="o", linewidth=2, label="NYUv2")
        ax.plot(x, diode_vals, marker="s", linewidth=2, label="DIODE")
        ax.set_xticks(x, groups, rotation=18)
        ax.set_title(f"Cross-dataset ablation on {metric}")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_tables() -> None:
    nyu_train = count_lines(ROOT / "data/nyuv2/bts_inputs/nyu_train_official.txt")
    nyu_test = count_lines(ROOT / "data/nyuv2/bts_inputs/nyu_test_official.txt")
    diode_train = count_lines(ROOT / "data/diode/bts_inputs/diode_train.txt")
    diode_val = count_lines(ROOT / "data/diode/bts_inputs/diode_val.txt")
    diode_test = count_lines(ROOT / "data/diode/bts_inputs/diode_test.txt")

    dataset_rows = [
        ["数据集", "训练集", "验证集", "测试集", "场景类型", "最大深度", "划分说明"],
        ["NYUv2", nyu_train, f"{nyu_test}（在线评估复用测试划分）", nyu_test, "室内", 10, "官方划分"],
        ["DIODE", diode_train, diode_val, diode_test, "室内/室外", 10, "本地划分文件"],
    ]
    write_csv(TABLE_DIR / "table_5_1_dataset_split_stats.csv", dataset_rows)
    write_text(
        TABLE_DIR / "table_5_1_dataset_split_stats.md",
        "\n".join(
            [
                "| 数据集 | 训练集 | 验证集 | 测试集 | 场景类型 | 最大深度 | 划分说明 |",
                "|---|---:|---:|---:|---|---:|---|",
                f"| NYUv2 | {nyu_train} | {nyu_test}（在线评估复用测试划分） | {nyu_test} | 室内 | 10 | 官方划分 |",
                f"| DIODE | {diode_train} | {diode_val} | {diode_test} | 室内/室外 | 10 | 本地划分文件 |",
            ]
        )
        + "\n",
    )

    exp_rows = [
        ["实验组名称", "数据集", "边缘分支", "边缘损失", "边缘权重", "实验目的"],
        ["baseline", "NYUv2 / DIODE", "否", "否", "0.00", "作为原始 BTS 参考基线"],
        ["branch only", "NYUv2 / DIODE", "是", "否", "0.00", "验证结构本身是否有效"],
        ["loss only", "NYUv2", "否（仅监督头）", "是", "0.05", "验证边缘监督本身是否有效"],
        ["loss only", "DIODE", "否（仅监督头）", "是", "0.02", "验证边缘监督本身是否有效"],
        ["full w=0.02", "NYUv2 / DIODE", "是", "是", "0.02", "验证较弱边缘约束的综合效果"],
        ["full w=0.05", "NYUv2 / DIODE", "是", "是", "0.05", "验证较强边缘约束的综合效果"],
        ["full w=0.10", "NYUv2", "是", "是", "0.10", "验证更强边缘监督是否过约束"],
    ]
    write_csv(TABLE_DIR / "table_5_3_experiment_config_summary.csv", exp_rows)
    write_text(
        TABLE_DIR / "table_5_3_experiment_config_summary.md",
        "\n".join(
            [
                "| 实验组名称 | 数据集 | 是否启用边缘分支 | 是否启用边缘损失 | 边缘权重 | 实验目的 |",
                "|---|---|---|---|---:|---|",
                "| baseline | NYUv2 / DIODE | 否 | 否 | 0.00 | 作为原始 BTS 参考基线 |",
                "| branch only | NYUv2 / DIODE | 是 | 否 | 0.00 | 验证结构本身是否有效 |",
                "| loss only | NYUv2 | 否（仅监督头） | 是 | 0.05 | 验证边缘监督本身是否有效 |",
                "| loss only | DIODE | 否（仅监督头） | 是 | 0.02 | 验证边缘监督本身是否有效 |",
                "| full w=0.02 | NYUv2 / DIODE | 是 | 是 | 0.02 | 验证较弱边缘约束的综合效果 |",
                "| full w=0.05 | NYUv2 / DIODE | 是 | 是 | 0.05 | 验证较强边缘约束的综合效果 |",
                "| full w=0.10 | NYUv2 | 是 | 是 | 0.10 | 验证更强边缘监督是否过约束 |",
            ]
        )
        + "\n",
    )


def generate_ai_prompts() -> None:
    prompts = {
        "fig_4_1_overall_architecture_prompt.md": """
请生成一张学术论文风格的网络结构图，白底、矢量感、中文标注清晰。
主题：Edge-Guided BTS 单目深度估计总体结构图。
必须包含：
1. 输入 RGB 图像；
2. DenseNet121 编码器；
3. BTS 解码器和三尺度局部平面指导分支：lpg8x8、lpg4x4、lpg2x2；
4. 浅层 skip0 特征分出一个轻量边缘分支；
5. 边缘分支经过卷积得到 edge logits，再经过 sigmoid 得到 edge probability；
6. edge probability 与末端 upconv1 特征发生乘性调制；
7. 调制后的特征与 reduc1x1、多尺度深度结果、edge probability 共同拼接；
8. 最终输出 depth map。
风格要求：结构清晰、箭头明确、模块颜色克制、适合课程报告插图。
""".strip(),
        "fig_4_2_bts_lpg_prompt.md": """
请生成一张学术论文风格原理图，白底、中文标注。
主题：原始 BTS 的多尺度局部平面指导原理图。
必须展示：
1. 编码器提取特征；
2. 解码器逐步上采样；
3. 在 8x8、4x4、2x2 三个尺度预测局部平面参数；
4. 从局部平面参数恢复各尺度深度图；
5. 多尺度深度逐级融合为最终深度图。
重点突出“局部平面参数 -> 深度恢复”的几何含义。
""".strip(),
        "fig_4_3_edge_fusion_prompt.md": """
请生成一张学术论文风格机制图，白底、中文标注。
主题：边缘分支与深度特征融合机制示意图。
必须展示：
1. skip0 特征进入边缘分支；
2. 得到 edge logits；
3. edge logits 经 sigmoid 变为 edge probability；
4. edge probability 对 upconv1 特征进行乘性调制 F' = F * (1 + P_edge)；
5. 调制后特征与 reduc1x1、lpg8x8、lpg4x4、lpg2x2、edge probability 一起拼接；
6. 进入最终卷积层输出深度图。
要求突出“边界位置被强化”的含义。
""".strip(),
        "fig_4_4_edge_target_prompt.md": """
请生成一张学术论文风格流程图，白底、中文标注。
主题：深度边缘监督目标构造示意图。
必须展示：
1. depth_gt 输入；
2. 局部深度梯度或相邻差分提取；
3. 阈值化得到候选边界；
4. 有效深度掩码约束；
5. 最终生成 edge_target；
6. edge logits 与 edge_target 进入 BCEWithLogitsLoss。
要求表达“由深度真值自动构造边缘监督目标”的逻辑。
""".strip(),
        "fig_3_1_problem_analysis_prompt.md": """
请生成一张课程报告用学术插图，白底、中文标注。
主题：原始单目深度估计在边界区域的典型误差示意图。
建议布局：
1. 左侧为输入 RGB 图像；
2. 中间为真实深度图；
3. 右侧为存在边界模糊的预测深度图；
4. 对桌角、门框、家具边缘等区域进行放大框标注；
5. 在放大框中强调“边界过平滑、前景背景混叠、深度跳变不清晰”。
整体风格要像实验分析图，不要科幻风，不要复杂背景。
""".strip(),
    }
    for name, content in prompts.items():
        write_text(PROMPT_DIR / name, content + "\n")


def generate_figures() -> None:
    plot_curve_comparison(
        dataset="NYUv2",
        baseline_train=ROOT / "outputs/logs/bts/nyu/bts_nyu_local_baseline/summaries",
        baseline_eval=ROOT / "outputs/logs/bts/nyu/eval/bts_nyu_local_baseline",
        improved_train=ROOT / "outputs/logs/bts/nyu/bts_nyu_edgeguided_w005/summaries",
        improved_eval=ROOT / "outputs/logs/bts/nyu/eval/bts_nyu_edgeguided_w005",
        output_path=FIG_DIR / "fig_7_1_nyu_training_eval_curves.png",
    )
    plot_curve_comparison(
        dataset="DIODE",
        baseline_train=ROOT / "outputs/logs/bts/diode/bts_diode_local_baseline/summaries",
        baseline_eval=ROOT / "outputs/logs/bts/diode/eval/bts_diode_local_baseline",
        improved_train=ROOT / "outputs/logs/bts/diode/bts_diode_edgeguided_w002/summaries",
        improved_eval=ROOT / "outputs/logs/bts/diode/eval/bts_diode_edgeguided_w002",
        output_path=FIG_DIR / "fig_7_2_diode_training_eval_curves.png",
    )
    make_visual_comparison(
        title="NYUv2 visual comparison: baseline vs edge-guided BTS",
        rgb_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_local_baseline/rgb",
        baseline_cmap_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_local_baseline/cmap",
        improved_cmap_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_edgeguided_w005/cmap",
        output_path=FIG_DIR / "fig_7_3_nyu_visual_comparison.png",
        normalizer=normalize_nyu_name,
    )
    make_visual_comparison(
        title="DIODE visual comparison: baseline vs edge-guided BTS",
        rgb_dir=ROOT / "outputs/results/bts/diode/bts_diode_local_baseline/rgb",
        baseline_cmap_dir=ROOT / "outputs/results/bts/diode/bts_diode_local_baseline/cmap",
        improved_cmap_dir=ROOT / "outputs/results/bts/diode/bts_diode_edgeguided_w002/cmap",
        output_path=FIG_DIR / "fig_7_4_diode_visual_comparison.png",
        normalizer=normalize_identity,
    )
    make_problem_analysis_figure(
        rgb_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_local_baseline/rgb",
        baseline_cmap_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_local_baseline/cmap",
        improved_cmap_dir=ROOT / "outputs/results/bts/nyu/bts_nyu_edgeguided_w005/cmap",
        output_path=FIG_DIR / "fig_3_1_problem_analysis.png",
    )
    plot_ablation_bars(
        dataset="NYUv2",
        groups=["baseline", "branch", "loss", "full-0.02", "full-0.05"],
        abs_rel=[0.168, 0.171, 0.163, 0.166, 0.166],
        rms=[0.547, 0.539, 0.543, 0.545, 0.537],
        output_path=FIG_DIR / "fig_8_1_nyu_ablation.png",
    )
    plot_ablation_bars(
        dataset="DIODE",
        groups=["baseline", "branch", "loss", "full-0.05", "full-0.02"],
        abs_rel=[0.542, 0.604, 0.540, 0.558, 0.516],
        rms=[1.773, 1.980, 1.754, 1.832, 1.741],
        output_path=FIG_DIR / "fig_8_2_diode_ablation.png",
    )
    plot_cross_dataset_ablation(FIG_DIR / "fig_8_3_cross_dataset_ablation.png")


def main() -> None:
    ensure_dirs()
    generate_tables()
    generate_ai_prompts()
    generate_figures()
    print("Generated assets under", ASSET_DIR)


if __name__ == "__main__":
    main()
