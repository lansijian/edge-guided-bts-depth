# 创新实现计划：Edge-Guided BTS

## 1. 实现目标

在不破坏当前 `BTS` baseline 可运行性的前提下，实现：

1. 轻量边缘分支
2. 边缘引导融合
3. 边缘损失
4. 新的训练配置与结果目录

## 2. 第一版最小可跑实现

第一版只做最必要的改动，保证尽快跑通。

### 2.1 模型改动

计划修改：

- `refs/bts/pytorch/bts.py`

拟增加：

1. `edge_extractor`
   - 从输入 RGB 提取浅层边缘特征
2. `edge_head`
   - 输出单通道边缘图
3. `edge_guided_fusion`
   - 在 `iconv1` 前后，将边缘特征与深度特征融合
4. 新输出
   - `pred_edge`

### 2.2 损失改动

计划修改：

- `refs/bts/pytorch/bts_main.py`

拟增加：

1. 从 GT 深度图生成边缘监督
2. 新增边缘损失项
3. 总损失：
   - `total_loss = silog_loss + lambda_edge * edge_loss`

### 2.3 配置改动

计划新增：

- `configs/bts/arguments_train_nyu_edgeguided.txt`
- `configs/bts/arguments_test_nyu_edgeguided.txt`
- `configs/bts/arguments_train_diode_edgeguided.txt`
- `configs/bts/arguments_test_diode_edgeguided.txt`

### 2.4 运行入口

计划新增：

- `src/run_bts_edgeguided_nyu_train.py`
- `src/run_bts_edgeguided_nyu_test.py`
- `src/run_bts_edgeguided_diode_train.py`
- `src/run_bts_edgeguided_diode_test.py`

## 3. 训练顺序建议

### 第一阶段

先在 `NYUv2` 上训练创新版：

1. 数据更稳定
2. baseline 更成熟
3. 便于快速验证模块是否有效

### 第二阶段

再迁移到 `DIODE`：

1. 复用相同模型结构
2. 调整少量参数
3. 完成跨数据集对比

## 4. 消融实现顺序

建议按以下顺序推进：

### 版本 A

- 原始 `BTS`

### 版本 B

- `BTS + edge loss`

说明：

- 先不改结构
- 只改损失
- 用于验证边缘监督本身是否有效

### 版本 C

- `BTS + edge branch`

说明：

- 加结构
- 不加边缘损失

### 版本 D

- `BTS + edge branch + edge loss`

说明：

- 作为最终创新版本

## 5. 实际执行顺序

1. 完成文档整理
2. 在 `bts.py` 中加入边缘分支与融合模块
3. 在 `bts_main.py` 中加入边缘损失
4. 新增训练配置与启动脚本
5. 完成最小冒烟测试
6. 完成 `NYUv2` 创新版训练与测试
7. 完成 `DIODE` 创新版训练与测试
8. 完成消融实验

## 6. 结果目录规划

建议新实验统一放在：

### NYUv2

- `outputs/logs/bts/nyu/bts_nyu_edgeguided/`
- `outputs/results/bts/nyu/bts_nyu_edgeguided/`

### DIODE

- `outputs/logs/bts/diode/bts_diode_edgeguided/`
- `outputs/results/bts/diode/bts_diode_edgeguided/`

## 7. 最终执行结论

当前已经完成：

1. baseline 选择
2. 创新方向确定
3. 创新文档建立
4. NYUv2 三轮创新实验
   - `w=0.10`
   - `w=0.02`
   - `w=0.05`
5. DIODE 两轮完整创新实验
   - `w=0.05`
   - `w=0.02`
6. `NYUv2` 与 `DIODE` 消融实验

当前最优版本：

- `NYUv2`：
  - `bts_nyu_edgeguided_w005`
- `DIODE`：
  - `bts_diode_edgeguided_w002`

最终结论：

1. `NYUv2` 上 `w=0.05` 最优
2. `DIODE` 上 `w=0.02` 最优
3. 消融实验说明创新收益主要来自 `edge loss` 与结构引导的联合

## 8. DIODE 创新阶段入口

### 8.1 训练入口

- `src/run_bts_edgeguided_diode_train_w005.py`
- `src/run_bts_edgeguided_diode_train_w002.py`

### 8.2 测试入口

- `src/run_bts_edgeguided_diode_test_w005.py`
- `src/run_bts_edgeguided_diode_test_w002.py`

### 8.3 配置文件

- `configs/bts/arguments_train_diode_edgeguided_w005.txt`
- `configs/bts/arguments_test_diode_edgeguided_w005.txt`
- `configs/bts/arguments_train_diode_edgeguided_w002.txt`
- `configs/bts/arguments_test_diode_edgeguided_w002.txt`

### 8.4 执行结论

1. `w=0.05` 已完成，但整体退化
2. `w=0.02` 已完成，并超过 `BTS baseline`
3. 因此 `DIODE` 最终采用 `bts_diode_edgeguided_w002`
