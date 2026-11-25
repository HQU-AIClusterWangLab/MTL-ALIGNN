# two：二维材料联合稳定性预测与可靠筛选

该项目针对本机路径 `D:\ck\two` 和数据库 `D:\ck\two\data\c2db.db` 编写，用于完成：

- Thermo、Dynamic和Joint稳定性预测；
- Random Forest、XGBoost、CGCNN-style和ALIGNN基线；
- Joint主任务的多任务学习；
- Deep Ensemble和MC Dropout不确定性评价；
- 校准、错误检测、risk–coverage和OOD压力测试；
- 概率排序与风险感知排序；
- `dyn_stab=Unknown`材料Top-100筛选和CIF导出。

## 已固定的实际环境

- Python：`D:\ck\anaconda_env\envs\two\python.exe`
- Python 3.10.20
- PyTorch 2.2.0+cu121
- DGL 2.2.1+cu121
- ALIGNN 2026.4.2
- ASE 3.28.0
- GPU：NVIDIA GeForce RTX 4090 D，23.99 GB
- 数据库：16,905条记录
- 字段：`uid`、`ehull`、`dyn_stab`
- 动力学标签：Yes 4,791；No 3,311；Unknown 8,803

本项目**不会自动安装或升级包**，以避免破坏已经验证可用的PyTorch/DGL/CUDA组合。

## 安装项目文件

1. 备份当前 `D:\ck\two` 中已有代码和结果。
2. 保留现有数据库：

```text
D:\ck\two\data\c2db.db
```

3. 将压缩包解压到 `D:\ck`，使代码目录最终为：

```text
D:\ck\two
```

压缩包不包含`c2db.db`，不会替换数据库。

## 标签定义

- Thermo：`ehull <= 0.10 eV/atom`
- Dynamic：`dyn_stab == "Yes"`
- Joint：Thermo和Dynamic同时为1
- 监督集合：`dyn_stab`为Yes或No
- Unknown池：`dyn_stab == "Unknown"`

`ehull`默认只用于标签和统计，并从主模型输入中排除。最终 Unknown 筛选按用户指定，
直接对全部 8,803 条材料预测并按验证集冻结后的校准 `pJ` 排名；
`ehull <= 0.10 eV/atom` 的 420 条子集只保留为事后诊断。补审中的 D+E_hull
部署基线是单独、显式标记的 Dynamic-only 对照，不与主模型混淆。

## VSCode运行

打开 `D:\ck\two`，按：

```text
Ctrl+Shift+P -> Tasks: Run Task
```

首次按顺序运行：

```text
00 - Environment, data and model smoke test
01 - Run unit tests
```

随后可分阶段运行：

```text
10 - Prepare all data and supervised graphs
20 - Run all main experiments
30 - Run all ablation experiments
40 - Screen Unknown pool and export Top-100
50 - Build review supplementary statistics
51 - Run review XGBoost deployment baselines
52 - Run complete review GNN matrix (long)
```

也可以运行：

```text
90 - Run all default experiments sequentially
```

默认连续任务**不运行OOD实验**。OOD代码通过以下任务单独启动：

```text
95 - Run optional brute-force OOD experiments
```

## PowerShell运行

```powershell
Set-ExecutionPolicy -Scope Process Bypass
cd D:\ck\two
.\RUN_SMOKE.ps1
.\RUN_DATA.ps1
.\RUN_MAIN.ps1
.\RUN_ABLATIONS.ps1
.\RUN_UNKNOWN.ps1
.\RUN_REVIEW_SUPPLEMENTS.ps1
.\RUN_REVIEW_BASELINES.ps1 -SkipGnn
```

一键连续运行：

```powershell
.\RUN_ALL.ps1
```

中断后可重新执行。神经网络训练使用`last_checkpoint.pt`断点续训，已完成任务通过`COMPLETED.json`自动跳过。

## 主实验与消融实验

主实验和消融实验的严格划分见：

```text
docs\01_experiment_map.md
```

简要分类：

### 主实验

- M01 Random Forest
- M02 XGBoost
- M03 CGCNN-style
- M04 Single-task ALIGNN
- M05 Equal-weight MTL ALIGNN
- M06 Full MTL ALIGNN，5成员Deep Ensemble
- M07 不确定性与校准
- M08 Joint融合比较
- M09 排序回溯
- M10 Unknown筛选

### 消融实验

- A01 Joint+Thermo
- A02 Joint+Dynamic
- A03 No logical consistency
- A04 Fixed task weights
- A05 Shuffled auxiliary labels
- A06 Adaptive weighting for all tasks
- A07 Gradient-conflict analysis

## 输出

主要结果：

```text
D:\ck\two\results\main
D:\ck\two\results\ablation
D:\ck\two\results\paper
D:\ck\two\results\review
D:\ck\two\results\review_supplements
```

Unknown Top-100和Materials Studio可读取CIF：

```text
D:\ck\two\results\main\M10_unknown_screening
```

其中包括：

- `probability_top100.xlsx`
- `risk_top100.xlsx`
- `unknown_screening_top100.xlsx`
- `probability_top100_cif\`
- `risk_top100_cif\`

补审汇总分别保存在 `results\review\summary.xlsx`（D-only 与 D+E_hull）和
`results\review_supplements\summary.xlsx`；后者的 `figures\` 同时提供 300-dpi PNG
与矢量 PDF。Ir₃BrI₂Te₃ 的 fail-closed DFT/phonon 输入准备目录位于
`dft\ir3bri2te3_phonon\`。

最终DFT候选由你基于Top-100排名、化学多样性和结构多样性在Materials Studio中人工选择。

## 关键严谨性规则

1. 测试集不参与阈值、温度、不确定性分数或lambda选择。
2. Unknown池不参与训练、验证、校准或模型选择。
3. 补审划分使用与元素顺序和超胞倍数无关的 canonical composition，并同时冻结
   random、composition-disjoint、chemical-system-disjoint 和
   structure-family-disjoint 四套 manifest；旧 `iid_split.csv` 不再称为严格组成隔离。
4. 精确重复结构发生已知标签冲突时全部排除。
5. Deep Ensemble温度缩放作用于每个成员logit，再平均概率。
6. `p_T*p_D`只称为乘积融合分数。
7. 不确定性标准差只表示模型预测分歧代理。
8. `p_J-lambda*u`只称为风险感知排序分数。
9. 风险排序是否有效由Precision@K、Yield@K和配对bootstrap验证。

更多方法学说明见：

```text
docs\02_methodological_guards.md
```

## 可选预训练基线

未提供预训练权重，因此默认流程从头训练。项目已提供可选配置：

```text
train\experiments\optional\O03_pretrained_alignn.yaml
pretrained\README.md
```

放入兼容权重后再单独运行，不包含在默认流程中。

## 补审实验与现实限制

严格划分与完整 GNN 矩阵见 `docs\review_matrix.md`。默认完整矩阵包含 120 个
ALIGNN seed runs，属于长时间 GPU 任务；运行器支持按 deployment/full-matrix 分批运行、
断点续训和完成标记跳过。Ir₃BrI₂Te₃ 的声子/soft-mode 实验还需要外部 DFT 引擎、
赝势及 phonon 工具，项目不会在缺少这些输入时伪造计算结果。
