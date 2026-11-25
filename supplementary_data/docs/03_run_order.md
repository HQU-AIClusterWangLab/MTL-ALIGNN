# 运行顺序

## 首次检查

在VSCode中运行任务：

1. `00 - Environment, data and model smoke test`
2. `01 - Run unit tests`

## 分阶段运行

1. `10 - Prepare all data and supervised graphs`
2. `20 - Run all main experiments`
3. `30 - Run all ablation experiments`
4. `40 - Screen Unknown pool and export Top-100`

## 一键运行

`90 - Run all default experiments sequentially`

该任务不运行OOD。训练中断后重新执行，已完成实验会跳过，未完成神经网络实验从`last_checkpoint.pt`恢复。

## 可选OOD

完成默认主实验后再运行：

`95 - Run optional brute-force OOD experiments`

## 预计计算量

默认流程包含多个随机种子和18个消融训练成员，完整执行可能持续较长时间。建议先完成Smoke、M04和M06单个种子检查，再启动连续任务。
