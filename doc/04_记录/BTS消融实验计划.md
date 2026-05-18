# BTS 消融实验记录

## 1. 目标

验证 `Edge-Guided BTS` 的性能提升分别来自哪一部分：

1. 边缘引导分支本身
2. 边缘监督损失本身
3. 边缘分支与边缘监督的组合

## 2. 消融设置

### 2.1 NYUv2

| 组别 | 模型名 | 结构开关 | 损失开关 | 权重 | 状态 |
|---|---|---|---|---:|---|
| baseline | `bts_nyu_local_baseline` | 关 | 关 | 0.00 | 已完成 |
| branch only | `bts_nyu_edgebranch_only` | `use_edge_guidance` | 关 | 0.00 | 已完成 |
| loss only | `bts_nyu_edgeloss_only_w005` | 关 | `use_edge_loss` | 0.05 | 已完成 |
| full | `bts_nyu_edgeguided_w005` | `use_edge_guidance` | 开 | 0.05 | 已完成 |
| full | `bts_nyu_edgeguided_w002` | `use_edge_guidance` | 开 | 0.02 | 已完成 |

### 2.2 DIODE

| 组别 | 模型名 | 结构开关 | 损失开关 | 权重 | 状态 |
|---|---|---|---|---:|---|
| baseline | `bts_diode_local_baseline` | 关 | 关 | 0.00 | 已完成 |
| branch only | `bts_diode_edgebranch_only` | `use_edge_guidance` | 关 | 0.00 | 已完成 |
| loss only | `bts_diode_edgeloss_only_w002` | 关 | `use_edge_loss` | 0.02 | 已完成 |
| full | `bts_diode_edgeguided_w002` | `use_edge_guidance` | 开 | 0.02 | 已完成 |
| full | `bts_diode_edgeguided_w005` | `use_edge_guidance` | 开 | 0.05 | 已完成 |

## 3. 结果汇总

### 3.1 NYUv2

| 组别 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16.256 | 0.168 | 0.069 | 0.547 | 0.126 | 0.200 | 0.763 | 0.947 | 0.987 |
| branch only | 16.108 | 0.171 | 0.069 | 0.539 | 0.128 | 0.199 | 0.761 | 0.947 | 0.987 |
| loss only | 16.145 | 0.163 | 0.069 | 0.543 | 0.119 | 0.199 | 0.759 | 0.948 | 0.989 |
| full w=0.02 | 16.143 | 0.166 | 0.069 | 0.545 | 0.123 | 0.199 | 0.765 | 0.949 | 0.987 |
| full w=0.05 | 16.077 | 0.166 | 0.068 | 0.537 | 0.123 | 0.197 | 0.768 | 0.950 | 0.988 |

### 3.2 DIODE

| 组别 | silog | abs_rel | log10 | rms | sq_rel | log_rms | d1 | d2 | d3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.348 | 0.542 | 0.160 | 1.773 | 1.444 | 0.458 | 0.411 | 0.710 | 0.864 |
| branch only | 47.239 | 0.604 | 0.178 | 1.980 | 1.866 | 0.609 | 0.389 | 0.682 | 0.842 |
| loss only | 32.651 | 0.540 | 0.157 | 1.754 | 1.487 | 0.454 | 0.431 | 0.720 | 0.868 |
| full w=0.05 | 33.359 | 0.558 | 0.167 | 1.832 | 1.460 | 0.472 | 0.388 | 0.672 | 0.853 |
| full w=0.02 | 32.192 | 0.516 | 0.155 | 1.741 | 1.344 | 0.448 | 0.433 | 0.707 | 0.873 |

## 4. 消融结论

### 4.1 NYUv2

1. `loss only` 已经能稳定带来收益
2. `branch only` 只有局部指标提升，整体不如 `loss only`
3. `full w=0.05` 取得最优结果

### 4.2 DIODE

1. `branch only` 会明显退化
2. `loss only` 基本接近 baseline，并在部分指标上更好
3. `full w=0.02` 才是有效配置
4. `full w=0.05` 约束过强

## 5. 代码与配置

### 5.1 代码改动

- `refs/bts/pytorch/bts.py`
- `refs/bts/pytorch/bts_main.py`
- `refs/bts/pytorch/bts_test.py`

新增独立开关：

- `--use_edge_guidance`
- `--use_edge_loss`

### 5.2 配置

- `configs/bts/arguments_train_nyu_edgebranch_only.txt`
- `configs/bts/arguments_test_nyu_edgebranch_only.txt`
- `configs/bts/arguments_train_nyu_edgeloss_only_w005.txt`
- `configs/bts/arguments_test_nyu_edgeloss_only_w005.txt`
- `configs/bts/arguments_train_diode_edgebranch_only.txt`
- `configs/bts/arguments_test_diode_edgebranch_only.txt`
- `configs/bts/arguments_train_diode_edgeloss_only_w002.txt`
- `configs/bts/arguments_test_diode_edgeloss_only_w002.txt`

### 5.3 脚本

- `src/run_bts_edgebranch_only_nyu_train.py`
- `src/run_bts_edgebranch_only_nyu_test.py`
- `src/run_bts_edgeloss_only_nyu_train_w005.py`
- `src/run_bts_edgeloss_only_nyu_test_w005.py`
- `src/run_bts_edgebranch_only_diode_train.py`
- `src/run_bts_edgebranch_only_diode_test.py`
- `src/run_bts_edgeloss_only_diode_train_w002.py`
- `src/run_bts_edgeloss_only_diode_test_w002.py`

## 6. 文档落点

- `doc/04_记录/运行记录.md`
- `doc/04_记录/NYUv2创新实验对比.md`
- `doc/04_记录/DIODE创新实验对比.md`
