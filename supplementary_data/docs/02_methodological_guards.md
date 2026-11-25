# 方法学保护规则

1. 主 Joint/MTL 模型不把 `ehull` 作为输入；只有显式命名的补审
   Dynamic+E_hull 部署基线使用连续 `ehull`，且不得把它与主模型混写。
2. `dyn_stab=Unknown`的数据在模型开发、阈值选择、校准和排序参数选择阶段完全隔离。
3. 旧 `iid_split.csv` 的 reduced-formula 口径经 canonical composition 复核后存在
   399 个跨集合组成组，因此只能作为 legacy IID 结果。补审比较使用
   `data/processed/review_splits` 中冻结的 random、composition-disjoint、
   chemical-system-disjoint 和 structure-family-disjoint manifest；审计结果以
   `audit.json` 和 `manifest_sha256.json` 为准。
4. 精确结构重复项先审计；已知标签冲突的重复结构全部排除，已知与Unknown重复时优先保留已知项。
5. Early stopping 只监控配置声明的验证集任务 PR-AUC：Joint 模型监控 Joint，
   Dynamic-only 部署基线监控 Dynamic。
6. 分类阈值只在验证集按MCC选择。
7. 温度和分类阈值只在与待评估 checkpoint/ensemble 对应的非测试验证集拟合，
   冻结后应用到该模型的测试集；不同模型或不同 ensemble size 之间不得转用概率标度。
8. Deep Ensemble 校准在同一 ensemble 内使用验证集拟合的共同温度缩放各成员
   logit 后再平均概率；5-member Unknown 推理必须使用 5-member 的温度和阈值。
9. 不确定性分数只在验证集按AURC选择。
10. 风险系数lambda只在验证集Thermo-positive子集选择。
11. 测试集只用于最终比较，不用于规则选择。
12. `p_T*p_D`仅是乘积融合分数，不解释为无条件成立的联合概率。
13. Ensemble/MC标准差是预测分歧代理，不是DFT物理误差条。
14. `p_J-lambda*u`是经验风险排序规则，不声称为具有覆盖保证的置信下界。
15. 按用户指定，最终 8,803 条 Unknown 全部直接获得校准 `pJ` 和排名；
    `ehull<=0.10` 子集只作诊断。该 Top-K 是 DFT 优先级列表，不称为已验证稳定材料。
