# MTL-ALIGNN: code, models, and supporting data

用于论文复核、模型推理与重新训练的精简材料包。文件从 `D:\ck\two` 和 `D:\ck\two dft` 复制后筛选，两个源目录保持原样。

## 目录

| 目录 | 内容 |
| --- | --- |
| `code/` | 数据准备、模型、训练、评价、绘图和测试代码 |
| `configs/` | 项目和各实验配置，包括原始配置 |
| `data/` | C2DB 数据库、标签、固定划分和预处理表格 |
| `environment/` | 原环境说明与本机已安装版本 |
| `licenses/` | 现有许可状态说明 |
| `models/` | 最终五成员 MTL-ALIGNN 的最佳权重，共 83.36 MB |
| `results/` | 主实验、基线、消融、OOD、补审、调参的预测表、指标和论文图表 |
| `supplementary_data/` | DFT 输入与最终输出、独有论文辅助材料、方法文档和来源清单 |

数据库 `data/c2db.db` 予以保留，共 16,905 条记录；源数据摘要中有监督样本为 8,102 条，Unknown 候选为 8,803 条。标签及既有预测值没有改写。

## 最终模型与论文结果

五个权重位于 `models/experiments/main/M06_full_mtl/seed_2025/` 至 `seed_2029/` 下的 `best_checkpoint.pt`。温度校准、阈值、排序参数、成员预测及集成结果仍位于 `results/main/M06_full_mtl/`。推理入口分别读取这两处，不再重复放置权重。

`results/main/M06_full_mtl/ensemble/ensemble_metrics.json` 的 `test_joint_raw.pr_auc` 为 `0.9063548999784637`，对应论文报告的 0.9064。Unknown 排序、Top-100 和 CIF 位于 `results/main/M10_unknown_screening/`。所有基线、消融和补审的现有预测表及指标保留，可直接用于结果复核；这些实验的权重需要按配置重新训练。

`results/paper/`、`results/paper_figures/`、`results/figure6/` 保留。旧论文汇总目录中的独有修订图、表和历史绘图代码仍位于 `supplementary_data/paper/`；重复文件删除后的对应位置记录在 `supplementary_data/provenance/DUPLICATE_MAP.csv`。旧汇总绘图脚本依赖历史目录布局，使用时应按该映射定位输入；主要绘图脚本位于 `code/scripts/`。

## 运行

本机示例使用既有环境；其他机器请先参考 `environment/` 配置兼容环境。

```powershell
cd D:\ck\MTL-ALIGNN
$PythonExe = 'D:\ck\anaconda_env\envs\two\python.exe'
& $PythonExe predict.py --dry-run
& $PythonExe train_fold.py --dry-run

# 首次实际推理：根据保留的数据库与配置重新生成 Unknown 图缓存
& $PythonExe predict.py --prepare-graphs

# 首次实际训练：重新生成有监督图缓存
& $PythonExe train_fold.py --prepare-graphs --seed 2025

# 表格基线训练不需要图缓存
& $PythonExe train_fold.py --experiment configs/experiments/main/M01_random_forest.yaml --backend tabular --seed 2025
```

图缓存 `data/processed/graphs/` 属于可生成文件，未包含在此精简包中。`--dry-run` 仅显示输入和缓存缺失情况；真正运行时，需要通过 `--prepare-graphs` 显式生成缓存。之后普通运行可复用已有缓存。生成缓存和重新训练会增加本地目录体积，已在 `.gitignore` 中排除。

`predict.py` 针对现有 Unknown 池运行五成员筛选。`train_fold.py` 按既有固定划分和随机种子运行，名称沿用示例目录格式，不表示 K 折交叉验证。新输出写入 `results/reproduced/`。`code/package_runtime.py` 在内存中重定位历史路径，保留的 CSV 和历史结果配置仍保持原始内容。原 PowerShell 和部分历史绘图脚本含本机绝对路径，运行前需检查。

## DFT 材料

`supplementary_data/dft/materials_studio/` 中保留 `.cell`、`.param`、结构文件、`.castep`、`.phonon`、收敛测试和其他文本输出；原项目的输入准备代码位于 `supplementary_data/dft/input_preparation/`。

为减小体积，已移除 `.castep_bin` 和 `.cst_esp` 大型二进制文件。此目录用于查看已有结果和从输入重新计算，不包含完整的中间状态续算文件。完整 Materials Studio 工程仍在未修改的源目录 `D:\ck\two dft` 中。

## 筛选与校验

已移除非最终模型权重、结构图缓存、运行日志、已有压缩包、编译缓存、编辑器设置和临时截图。`review - 副本` 与旧论文汇总的重复文件逐一按 SHA-256 确认存在保留副本后再删除；独有论文图表和数据保留。

`supplementary_data/provenance/FILE_INDEX.csv` 记录目前仍保留的源文件及原路径。`DELETED_FILES.csv` 记录删除文件、校验值、原因和源位置；`DUPLICATE_MAP.csv` 标明重复文件的保留位置。`MODEL_INDEX.csv` 列出五个最终权重；`CHECKSUMS.sha256` 和 `PACKAGE_CHECKSUMS.sha256` 为精简后的当前校验清单；`CLEANUP_REPORT.json` 记录数量、体积和验证结果。初次完整复制的历史报告仍予保留。

整理及精简没有重新训练模型或重新运行整池推理和 DFT。原目录未提供项目许可证，当前 LICENSE 仅记录许可状态，未擅自指定 MIT 或其他开源许可。
