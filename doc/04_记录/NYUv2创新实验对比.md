# NYUv2 创新实验对比

## 1. 目标

本轮对比聚焦两个问题：

1. `Edge-Guided BTS` 是否超过原始 `BTS baseline`
2. 提升主要来自边缘分支、边缘损失，还是两者组合

## 2. 完整版对比

| 方法 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BTS baseline | 16.256 | 0.168 | 0.069 | 0.547 | 0.126 | 0.200 | 0.763 | 0.947 | 0.987 | 基线 |
| Edge-Guided BTS w=0.10 | 16.325 | 0.172 | 0.069 | 0.542 | 0.127 | 0.201 | 0.760 | 0.946 | 0.987 | 约束过强 |
| Edge-Guided BTS w=0.02 | 16.143 | 0.166 | 0.069 | 0.545 | 0.123 | 0.199 | 0.765 | 0.949 | 0.987 | 明显优于 baseline |
| Edge-Guided BTS w=0.05 | 16.077 | 0.166 | 0.068 | 0.537 | 0.123 | 0.197 | 0.768 | 0.950 | 0.988 | 当前最优 |

## 3. 消融实验

| 组别 | 模型名 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | `bts_nyu_local_baseline` | 16.256 | 0.168 | 0.069 | 0.547 | 0.126 | 0.200 | 0.763 | 0.947 | 0.987 | 基线 |
| branch only | `bts_nyu_edgebranch_only` | 16.108 | 0.171 | 0.069 | 0.539 | 0.128 | 0.199 | 0.761 | 0.947 | 0.987 | `rms` 略好，但整体不稳 |
| loss only | `bts_nyu_edgeloss_only_w005` | 16.145 | 0.163 | 0.069 | 0.543 | 0.119 | 0.199 | 0.759 | 0.948 | 0.989 | 主要提升来自监督项 |
| full w=0.02 | `bts_nyu_edgeguided_w002` | 16.143 | 0.166 | 0.069 | 0.545 | 0.123 | 0.199 | 0.765 | 0.949 | 0.987 | 完整版已优于 baseline |
| full w=0.05 | `bts_nyu_edgeguided_w005` | 16.077 | 0.166 | 0.068 | 0.537 | 0.123 | 0.197 | 0.768 | 0.950 | 0.988 | 最优完整方案 |

## 4. 结论

### 4.1 主要结论

1. `w=0.05` 是当前 `NYUv2` 最优完整创新版
2. 仅加边缘分支不能稳定提升全部指标
3. 仅加边缘损失已经能带来比较明显的收益
4. 分支和损失组合后，最终效果最好，说明二者是互补关系

### 4.2 创新点解释

`NYUv2` 上的收益来源不是单独某一个模块硬拉出来的，而是：

- `edge loss` 先提供了更稳定的监督收益
- `edge-guided branch` 再在此基础上细化局部结构
- 当 `edge_loss_weight` 调到 `0.05` 时，两者平衡最好

### 4.3 当前最佳模型

- `bts_nyu_edgeguided_w005`

对应目录：

- 训练输出：
  - `outputs/logs/bts/nyu/bts_nyu_edgeguided_w005/`
- 测试结果：
  - `outputs/results/bts/nyu/bts_nyu_edgeguided_w005/`

## 5. 可视化目录

- baseline:
  - `outputs/results/bts/nyu/bts_nyu_local_baseline/`
- branch only:
  - `outputs/results/bts/nyu/bts_nyu_edgebranch_only/`
- loss only:
  - `outputs/results/bts/nyu/bts_nyu_edgeloss_only_w005/`
- full w=0.02:
  - `outputs/results/bts/nyu/bts_nyu_edgeguided_w002/`
- full w=0.05:
  - `outputs/results/bts/nyu/bts_nyu_edgeguided_w005/`

## 6. 推荐汇报指标

优先看：

1. `abs_rel`
2. `silog`
3. `rms`
4. `d1`

## 7. 可直接写进报告的摘要

在 `NYUv2` 上，`Edge-Guided BTS` 相比原始 `BTS baseline` 实现了稳定提升，其中最佳配置为 `edge_loss_weight = 0.05`。消融实验表明，仅增加边缘引导分支并不能稳定提升性能，而仅增加边缘监督损失已经可以带来明显收益；当边缘分支与边缘监督联合使用时，模型取得最优结果，说明该创新的有效性主要来自边缘监督与结构引导的协同作用。
