# 主实验与消融实验

## 主实验

| 编号 | 内容 | 直接回答的问题 |
|---|---|---|
| M01 | Random Forest | 手工组成/结构特征基线能达到什么水平 |
| M02 | XGBoost | 强树模型基线能否替代晶体图模型 |
| M03 | CGCNN-style | 常规晶体图卷积与ALIGNN的差异 |
| M04 | Single-task ALIGNN | 多任务学习相对单任务是否有收益 |
| M05 | Equal-weight MTL ALIGNN | 普通多任务建模是否有效 |
| M06 | Full MTL ALIGNN，5个种子 | 完整模型的分类性能和Deep Ensemble成员 |
| M07 | Single/MC Dropout/Deep Ensemble | 不确定性、校准、错误检测和选择性预测 |
| M08 | Direct/Product/Minimum/Learned fusion | 单独Joint头是否必要 |
| M09 | Probability与risk-aware ranking | 风险排序是否提高固定预算筛选收益 |
| M10 | Unknown Top-100 | 最终候选输出与Materials Studio CIF导出 |

## 消融实验

所有消融共享同一数据划分、训练超参数和随机种子，原则上只改变一个因素。

| 编号 | 相对完整模型的唯一变化 |
|---|---|
| A01 | 移除Dynamic辅助任务，只保留Joint+Thermo |
| A02 | 移除Thermo辅助任务，只保留Joint+Dynamic |
| A03 | 删除逻辑一致性损失 |
| A04 | 自适应辅助权重改为固定权重 |
| A05 | 在无逻辑约束对照中，仅打乱训练集Thermo和Dynamic辅助标签 |
| A06 | Joint也参与任务级自适应加权 |
| A07 | 不训练新模型；计算共享编码器任务梯度余弦 |

## 默认流程之外

`RUN_OOD_OPTIONAL.ps1`会生成chemical-system、structure-family、layer-group和element holdout定义。默认`RUN_ALL.ps1`不会启动OOD训练。
