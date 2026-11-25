# 补审严格划分实验矩阵

本矩阵把四种数据划分与两组不同的审稿问题分开处理。所有 review 配置均固定使用
`[2025, 2026, 2027, 2028, 2029]` 五个种子；同一划分内的模型共享完全相同的
train/validation/test manifest，因此可以按种子配对比较。

## 数据划分

| 简称 | manifest | 约束 |
|---|---|---|
| random | `review_splits/random_iid_stratified.csv` | 分层随机 IID 对照 |
| composition | `review_splits/composition_disjoint.csv` | 规范化化学组成跨集合不重叠 |
| chemical | `review_splits/chemical_system_disjoint.csv` | 元素集合跨集合不重叠 |
| structure | `review_splits/structure_family_disjoint.csv` | 结构家族键跨集合不重叠 |

运行器会先执行 `data/make_review_splits.py`，重新生成并审计这些 manifest。公平比较时，
不得把旧的 `iid_split.csv` 结果和新的严格 manifest 结果当作同一划分下的配对实验。

## Deployment 基线：R01--R08

每个划分包含两个以 `dynamic` 为唯一预测目标、以验证集 dynamic PR-AUC 早停的模型：

- D-only：结构图输入；
- D+E_hull：结构图输入加仅由训练集拟合标准化统计量的连续 `ehull` 条件特征。

这 8 个配置共计 40 个 GNN seed runs。它们回答部署阶段“已知/可用 E_hull 是否改善动力学稳定性预测”的问题，不应与以下 joint 标签模型直接混用指标。

## Joint-stability 完整矩阵：R21--R36

| 划分 | Single（joint-only） | Equal MTL | Full MTL | Full-no-logic |
|---|---:|---:|---:|---:|
| random | R21 | R22 | R23 | R24 |
| composition | R25 | R26 | R27 | R28 |
| chemical | R29 | R30 | R31 | R32 |
| structure | R33 | R34 | R35 | R36 |

参数严格复用既有模板：Single 对应 M04，Equal MTL 对应 M05，Full MTL 对应 M06，
Full-no-logic 对应 A03。除实验名、结果类别、split manifest 与统一五种子外，不改变
模型规模、优化器和训练超参数。Full 与 Full-no-logic 仅改变逻辑正则项，适合直接做
配对消融；Single、Equal、Full 则用于区分多任务学习和自适应辅助权重的贡献。

完整矩阵为 16 个配置、80 个 GNN seed runs。

## 运行选择

从项目根目录执行：

```powershell
# 默认：XGBoost deployment + 全部 GNN（R01--R08 与 R21--R36）
.\RUN_REVIEW_BASELINES.ps1

# 只跑 deployment GNN（R01--R08），跳过 XGBoost
.\RUN_REVIEW_BASELINES.ps1 -SkipTabular -GnnSet Deployment

# 只跑 joint-stability 完整 GNN 矩阵（R21--R36）
.\RUN_REVIEW_BASELINES.ps1 -SkipTabular -GnnSet FullMatrix

# 只跑 XGBoost deployment 基线
.\RUN_REVIEW_BASELINES.ps1 -SkipGnn
```

默认调用量为 8 个 XGBoost 配置（40 seed runs）加 24 个 GNN 配置（120 seed runs），
合计 160 seed runs。`-Force` 会传给每个训练入口；不加该参数时，已有完成标记的实验
按项目现有跳过规则处理。实际墙钟时间取决于图缓存、显卡和早停轮数，建议先用
`-GnnSet Deployment` 或 `-GnnSet FullMatrix` 分批占用 GPU，并保留每个种子的独立输出。
