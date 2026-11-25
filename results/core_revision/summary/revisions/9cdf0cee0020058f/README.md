# 固定 IID 核心补实验

状态：部分完成；缺失训练见下文。只使用既有 iid_split.csv，未重新划分数据。

## 表格回答的问题

- main_5seed_table.csv：四种方法在相同 seeds 上的 individual-run 原始概率指标 mean±SD（ddof=1）；缺 seed 明确标记，不能称为完整五种子结论。集成指标另见 01_main_5seed/ensemble_metrics.csv。
- joint_construction_table.csv：直接 Joint 与独立 T-only/D-only 集成的 Hard AND、Product、Minimum 比较。Hard AND 不计算 AP/ROC-AUC。
- paired_bootstrap_table.csv：Full MTL 减替代方法的 AP；材料 ID 严格配对，2000 次 bootstrap、95% percentile CI。双侧符号尾概率是描述性 bootstrap 量，未作多重检验校正，不能单独解释为确认性显著性检验。
- beta_sensitivity_table.csv：beta 与 AP/MCC/逻辑违反率的关系；核心使用 raw，calibrated 单列；缺失点不插值，不按 test 选择 beta。
- screening_efficiency_table.csv：找回 50%/80%/90% 真正阳性所需候选验证比例。完整曲线、10/20/30% enrichment 与材料 bootstrap CI 见 04_screening_efficiency。它表示候选数量预算，不代表真实 DFT 耗时节省。
- risk_increment_table.csv：任务分解风险对错误检测和选择性风险是否提供额外信息；保留所有方法，无预设排序结论。

## 复用与新增

已复用的来源逐 seed 记录在 configuration_audit.csv 与 manifest。M04 与 Q1 Joint 合并前检查 scientific config；缺省 monitor_task=joint 按代码默认值归一。A03 的 beta=0 使逻辑分支完全不执行，因此 lower_bound 标志差异不改变损失；beta 配置本身保持 lower_bound=true，只更改 beta。既有 bootstrap、temperature、ensemble、risk-coverage 实现继续使用。

Equal+LC 与未完成 beta 需要新训练；其它完成且合格的模型跳过。Joint 构造、screening、risk、bootstrap 均是既有冻结预测的后处理。新训练实际来源位于 core_revision/training；当前缺失 job：

- Equal + LC，seed=2025
- Equal + LC，seed=2026
- Equal + LC，seed=2027
- Equal + LC，seed=2028
- Equal + LC，seed=2029
- beta=0.01，seed=2025
- beta=0.01，seed=2026
- beta=0.01，seed=2027
- beta=0.05，seed=2025
- beta=0.05，seed=2026
- beta=0.05，seed=2027
- beta=0.20，seed=2025
- beta=0.20，seed=2026
- beta=0.20，seed=2027
- beta=0.50，seed=2025
- beta=0.50，seed=2026
- beta=0.50，seed=2027

## 防泄漏及溯源

温度、阈值、风险标准化与 LogisticRegression 只拟合 validation；test 仅用于冻结后的评估和 bootstrap。风险模型固定 C=1、lbfgs、random_state=20260917，没有 test 调参。主集成和风险分析统一 calibrated 概率；成员摘要和 beta 核心用 raw，明确分表。无成功 seed 筛选。

历史结果未保存训练当时的代码/数据内容哈希，不能追溯证明当时逐字节代码一致；当前记录的完整 resolved config、环境、实际 batch size、split、材料与标签已检查。新结果保存 run_contract 和当前代码/数据指纹。

## 运行与结果版本

VS Code 04 为预检查；05 为完整 CUDA 队列；06 为仅分析。重复调用跳过相同输入的已完成分析。新增 seed 后旧科学文件保留，新版本写入各分析目录 revisions/<signature>/；最新路径以 _runner/status.json 的 analysis 与本 result_manifest.json 为准。训练故障会立即停止并记录 failed_job，原故障日志保留。文献模板为空时正常退出，不自动造文献。
