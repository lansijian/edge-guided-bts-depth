# 基于边缘引导细化的 BTS 单目深度估计改进

本项目是《机器视觉设计与实践》课程设计仓库，围绕单目深度估计任务开展了以下工作：

- 复现 `BTS` 与 `GuidedDecoding` 基线
- 在 `BTS` 基础上实现 `Edge-Guided BTS` 改进方法
- 在 `NYUv2` 与 `DIODE` 数据集上完成对比实验与消融实验
- 整理课程报告、实验记录、结果图表与可复现实验脚本

当前仓库已经包含：

- 课程报告正文 Markdown 版
- 报告所需图表与结构示意图
- 数据准备、训练、测试、评估脚本
- 实验日志、可视化结果与配置文件

## 1. 项目概览

本文方法针对原始 BTS 在边界区域容易出现深度过平滑、轮廓模糊和前后景混叠的问题，在解码末端引入轻量边缘分支，并加入边缘监督损失与边缘引导融合机制，以提升深度突变区域的建模能力。

核心实验结论如下：

- `NYUv2` 最优模型：`bts_nyu_edgeguided_w005`
- `DIODE` 最优模型：`bts_diode_edgeguided_w002`
- 边缘监督损失是更稳定的收益来源
- 边缘引导结构在合适权重下可进一步提升结果

## 2. 目录结构

建议按下面的方式理解整个仓库：

```text
.
├─ README.md                             # GitHub 首页说明
├─ 机器视觉设计与实践.docx                # 课程设计文档版
├─ 机器视觉课程设计26.pdf                 # 课程设计 PDF 版
├─ 封面.pdf                               # 封面文件
├─ checkpoints/                          # 预留的权重存放目录
├─ configs/
│  └─ bts/                               # BTS 训练/测试参数配置
├─ data/
│  ├─ nyuv2/                             # NYUv2 数据与划分文件
│  └─ diode/                             # DIODE 数据与划分文件
├─ doc/
│  ├─ 01_总览/                            # 项目说明、实验规划
│  ├─ 02_准备/                            # 环境与数据准备记录
│  ├─ 03_实现/                            # 代码适配与创新实现说明
│  └─ 04_记录/                            # 实验对比与运行记录
├─ outputs/
│  ├─ logs/                              # 训练日志、checkpoint、TensorBoard
│  └─ results/                           # 测试输出、可视化结果、评估结果
├─ refs/                                 # 引用的开源仓库副本
│  ├─ bts/
│  ├─ GuidedDecoding/
│  ├─ ZoeDepth/
│  └─ fast-depth/
├─ report/
│  ├─ 课程设计报告正文初稿.md              # 报告正文 Markdown 版
│  └─ assets/
│     ├─ figures/                        # 报告使用图片
│     ├─ tables/                         # 报告使用表格
│     └─ ai_prompts/                     # 结构图生成提示词
└─ src/
   ├─ 数据准备脚本
   ├─ BTS 训练/测试脚本
   ├─ GuidedDecoding 运行脚本
   ├─ smoke test 脚本
   └─ generate_report_assets.py          # 生成报告图表与表格
```

## 3. 关键文件说明

### 3.1 报告与图表

- 报告正文：
  - [report/课程设计报告正文初稿.md](report/课程设计报告正文初稿.md)
- 报告图片目录：
  - [report/assets/figures](report/assets/figures)
- 报告表格目录：
  - [report/assets/tables](report/assets/tables)

目前报告中使用的主要图表已经齐全，包括：

- `图 3-1` 问题分析图
- `图 4-1` ~ `图 4-4` 结构与原理图
- `图 7-1`、`图 7-2` 训练与在线评估曲线
- `图 7-3`、`图 7-4` 定性可视化对比图
- `图 8-1`、`图 8-2` 单数据集消融图
- `图 8-3` 跨数据集消融综合图

### 3.2 代码与脚本

`src/` 下脚本大致可以分为四类：

1. 数据准备脚本
   - `convert_nyuv2.py`
   - `prepare_diode.py`
   - `generate_bts_filelists.py`

2. BTS 训练与测试脚本
   - `run_bts_edgeguided_*`
   - `run_bts_edgebranch_only_*`
   - `run_bts_edgeloss_only_*`

3. GuidedDecoding 基线脚本
   - `run_guideddecoding_nyu_baseline.py`
   - `run_guideddecoding_nyu_eval.py`
   - `run_guideddecoding_diode_baseline.py`
   - `run_guideddecoding_diode_eval.py`

4. 辅助脚本
   - `smoke_test_*`
   - `generate_report_assets.py`

### 3.3 配置文件

`configs/bts/` 中保存了各实验组的参数文件，覆盖：

- `baseline`
- `edgebranch_only`
- `edgeloss_only`
- `edgeguided_w002`
- `edgeguided_w005`
- `edgeguided_w010`（NYUv2）

训练与测试参数分别存放为 `arguments_train_*.txt` 和 `arguments_test_*.txt`。

## 4. 实验结果摘要

### 4.1 GuidedDecoding baseline

`NYUv2`

- `RMSE = 1.195`
- `MAE = 0.853`
- `REL = 1.002`
- `Lg10 = 0.163`
- `Delta1 = 0.474`
- `Delta2 = 0.742`
- `Delta3 = 0.870`

`DIODE`

- `RMSE = 7.693`
- `MAE = 5.281`
- `REL = 5.063`
- `Lg10 = 0.260`
- `Delta1 = 0.317`
- `Delta2 = 0.523`
- `Delta3 = 0.680`

### 4.2 Edge-Guided BTS 最优结果

`NYUv2`

- `silog = 16.077`
- `abs_rel = 0.166`
- `log10 = 0.068`
- `rms = 0.537`
- `sq_rel = 0.123`
- `log_rms = 0.197`
- `d1 = 0.768`
- `d2 = 0.950`
- `d3 = 0.988`

`DIODE`

- `silog = 32.192`
- `abs_rel = 0.516`
- `log10 = 0.155`
- `rms = 1.741`
- `sq_rel = 1.344`
- `log_rms = 0.448`
- `d1 = 0.433`
- `d2 = 0.707`
- `d3 = 0.873`

## 5. 运行环境

本项目主要在以下环境中完成：

- Python 环境：`D:\Anaconda_envs\PulseWeave`
- 主要依赖：`PyTorch`、`torchvision`

如果需要在新环境中复现，建议优先保证以下几点：

1. `PyTorch` 与 CUDA 版本匹配
2. 数据集目录与划分文件路径一致
3. `TORCH_HOME` 等缓存目录已正确设置
4. `outputs/` 具有写权限

## 6. 常用运行入口

### 6.1 GuidedDecoding baseline

```powershell
D:\Anaconda_envs\PulseWeave\python.exe src\run_guideddecoding_nyu_baseline.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_guideddecoding_nyu_eval.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_guideddecoding_diode_baseline.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_guideddecoding_diode_eval.py
```

### 6.2 BTS 改进模型

按实验组分别运行 `src/` 下对应脚本，例如：

```powershell
D:\Anaconda_envs\PulseWeave\python.exe src\run_bts_edgeguided_nyu_train_w005.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_bts_edgeguided_nyu_test_w005.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_bts_edgeguided_diode_train_w002.py
D:\Anaconda_envs\PulseWeave\python.exe src\run_bts_edgeguided_diode_test_w002.py
```

### 6.3 报告图表重新生成

```powershell
D:\Anaconda_envs\PulseWeave\python.exe src\generate_report_assets.py
```

## 7. 文档索引

`doc/` 目录下保留了课程设计过程文档，建议按下面顺序阅读：

### 7.1 总览

- [doc/01_总览/项目说明.md](doc/01_总览/项目说明.md)
- [doc/01_总览/实验规划.md](doc/01_总览/实验规划.md)

### 7.2 准备

- [doc/02_准备/环境与数据准备.md](doc/02_准备/环境与数据准备.md)

### 7.3 实现

- [doc/03_实现/开源代码来源.md](doc/03_实现/开源代码来源.md)
- [doc/03_实现/对比仓库适配说明.md](doc/03_实现/对比仓库适配说明.md)
- [doc/03_实现/代码适配与修改记录.md](doc/03_实现/代码适配与修改记录.md)
- [doc/03_实现/创新方案_EdgeGuided_BTS.md](doc/03_实现/创新方案_EdgeGuided_BTS.md)
- [doc/03_实现/创新实现计划_EdgeGuided_BTS.md](doc/03_实现/创新实现计划_EdgeGuided_BTS.md)

### 7.4 记录

- [doc/04_记录/运行记录.md](doc/04_记录/运行记录.md)
- [doc/04_记录/NYUv2创新实验对比.md](doc/04_记录/NYUv2创新实验对比.md)
- [doc/04_记录/DIODE创新实验对比.md](doc/04_记录/DIODE创新实验对比.md)
- [doc/04_记录/BTS消融实验计划.md](doc/04_记录/BTS消融实验计划.md)

## 8. GitHub 上传建议

如果准备上传到 GitHub，推荐保留以下内容：

- `src/`
- `configs/`
- `doc/`
- `report/`
- `README.md`
- 课程报告导出文件（`pdf/docx`）

如果仓库体积过大，可按需要裁剪以下内容：

- `data/` 中的大体积原始数据
- `outputs/cache/` 中的模型缓存
- `outputs/logs/` 中非关键中间 checkpoint
- `outputs/results/` 中体积较大的重复可视化文件
- `refs/` 中完整开源仓库副本

更稳妥的做法是：

1. 代码、配置、报告与关键结果图上传 GitHub
2. 大模型权重与完整数据集改用网盘或 Release 附件
3. 在仓库主页保留目录说明与复现入口

## 9. 当前状态

当前仓库已经具备较完整的课程设计交付形态：

- 代码可运行
- 结果可追溯
- 图表可复用
- 报告可继续润色或直接导出

如果后续还要继续整理，优先建议做两件事：

1. 增加 `.gitignore`，过滤缓存、临时日志和大文件
2. 统一部分中文图名与英文图名，便于长期维护和跨平台协作
