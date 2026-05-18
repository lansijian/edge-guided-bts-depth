# DIODE 创新实验对比

## 1. 目标

验证 `Edge-Guided BTS` 在 `DIODE` 上是否超过原始 `BTS baseline`，并通过消融确定提升来源。

## 2. 完整版对比

| 方法 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BTS baseline | 32.348 | 0.542 | 0.160 | 1.773 | 1.444 | 0.458 | 0.411 | 0.710 | 0.864 | 基线 |
| Edge-Guided BTS w=0.05 | 33.359 | 0.558 | 0.167 | 1.832 | 1.460 | 0.472 | 0.388 | 0.672 | 0.853 | 整体退化 |
| Edge-Guided BTS w=0.02 | 32.192 | 0.516 | 0.155 | 1.741 | 1.344 | 0.448 | 0.433 | 0.707 | 0.873 | 当前最优 |

## 3. 消融实验

| 组别 | 模型名 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | `bts_diode_local_baseline` | 32.348 | 0.542 | 0.160 | 1.773 | 1.444 | 0.458 | 0.411 | 0.710 | 0.864 | 基线 |
| branch only | `bts_diode_edgebranch_only` | 47.239 | 0.604 | 0.178 | 1.980 | 1.866 | 0.609 | 0.389 | 0.682 | 0.842 | 明显退化 |
| loss only | `bts_diode_edgeloss_only_w002` | 32.651 | 0.540 | 0.157 | 1.754 | 1.487 | 0.454 | 0.431 | 0.720 | 0.868 | 接近并部分优于 baseline |
| full w=0.05 | `bts_diode_edgeguided_w005` | 33.359 | 0.558 | 0.167 | 1.832 | 1.460 | 0.472 | 0.388 | 0.672 | 0.853 | 约束过强 |
| full w=0.02 | `bts_diode_edgeguided_w002` | 32.192 | 0.516 | 0.155 | 1.741 | 1.344 | 0.448 | 0.433 | 0.707 | 0.873 | 最优完整方案 |

## 4. 结论

### 4.1 主要结论

1. `w=0.02` 是当前 `DIODE` 最优完整创新版
2. 仅加边缘分支会明显破坏 `DIODE` 表现
3. 仅加边缘损失已经接近并部分超过 baseline
4. 完整版只有在较小权重 `0.02` 下才有效，`0.05` 会过强

### 4.2 创新点解释

`DIODE` 的场景变化更大、边缘噪声更复杂，因此：

- 没有监督约束的边缘分支会引入明显干扰
- 仅靠边缘监督可以带来一定正向收益
- 分支与监督组合后，必须使用更小的权重 `0.02` 才能稳定工作

这说明该创新在 `DIODE` 上依赖更谨慎的正则强度控制。

### 4.3 当前最佳模型

- `bts_diode_edgeguided_w002`

对应目录：

- 训练输出：
  - `outputs/logs/bts/diode/bts_diode_edgeguided_w002/`
- 测试结果：
  - `outputs/results/bts/diode/bts_diode_edgeguided_w002/`

## 5. 可视化目录

- baseline:
  - `outputs/results/bts/diode/bts_diode_local_baseline/`
- branch only:
  - `outputs/results/bts/diode/bts_diode_edgebranch_only/`
- loss only:
  - `outputs/results/bts/diode/bts_diode_edgeloss_only_w002/`
- full w=0.05:
  - `outputs/results/bts/diode/bts_diode_edgeguided_w005/`
- full w=0.02:
  - `outputs/results/bts/diode/bts_diode_edgeguided_w002/`

## 6. 可直接写进报告的摘要

在 `DIODE` 上，`Edge-Guided BTS` 的最佳配置为 `edge_loss_weight = 0.02`。消融实验显示，单独加入边缘引导分支会导致性能明显下降，而单独加入边缘监督损失则能够带来接近基线甚至部分更优的结果；当边缘分支与边缘监督联合使用且边缘损失权重控制在较小范围时，模型能够超过原始 `BTS baseline`。这说明该创新在复杂场景数据集上是有效的，但对边缘约束强度更敏感。
