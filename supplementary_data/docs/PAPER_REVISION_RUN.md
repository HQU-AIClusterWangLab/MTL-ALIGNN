# 论文补充实验运行说明

## 现在运行哪一个任务

在 VS Code 中打开整个 `D:\ck\two` 文件夹，按 **Ctrl+Shift+P**，选择 **Tasks: Run Task / 任务: 运行任务**，再选择：

**`60 - Paper revision - pilot 18 runs (resume)`**

这个任务会先检查环境、固定划分、图缓存与已有基线，然后顺序完成 18 个训练，最后自动校准、评估和汇总。无需先跑数据准备任务。若 VS Code 没刷新任务列表，执行 `Developer: Reload Window`。

训练期间可在任务终端查看当前实验、seed 和 epoch。中断后再次运行同一个 60 号任务：已经完成的成员自动跳过，未完成成员从最后一个完整 epoch 的检查点恢复。当前 epoch 中途停止的进度会重新计算。

## 18 次具体补什么

固定使用现有 composition-disjoint 和 structure-family-disjoint 两种划分，各用 2025、2026、2027 三个种子。

| 对照 | 实际训练内容 | 每种划分次数 | 用途 |
|---|---|---:|---|
| Equal+LC | T、D、J 三个 BCE，权重严格均为 1，逻辑约束系数 0.10 | 3 | 与 Equal、Full、Full-noLC 组成权重策略 × 逻辑约束的对照 |
| Four-state | 00、01、10、11 四状态的一个 softmax 头，仅四分类交叉熵 | 3 | 检查显式联合分布是否优于软逻辑约束 |
| T+D-product | 共用 ALIGNN 编码器，仅训练 T、D 两个 BCE；每个成员的 J=pT×pD | 3 | 检查独立联合预测头是否必要 |

总计 **3 种方法 × 2 种划分 × 3 个种子 = 18 次**。编码器、学习率、batch size、最多 200 epochs、早停 patience=30 与已有严格划分基线匹配。结构族划分使用项目已有的结构族标识，其严格程度取决于该分组定义，不等同于完整的晶体结构相似性排重。

三种模型都沿用验证集 Joint AP 选择 checkpoint。因此 T+D-product 的准确表述是“没有联合任务训练损失”，不能写成“整个开发过程完全不使用联合标签”。产品规则事先固定，不依据测试集在 product/min/其他融合方式中择优。

四状态模型对原生四分类 logits 拟合一个共享温度，再推导 T、D、J；T+D-product 分别校准 T、D，再计算每个成员的乘积，最后平均。两者校准后仍满足联合概率边界。Equal+LC 与历史三头基线按原先方法分别校准各头，同时报告由此产生的逻辑违反率，不做事后裁剪。

## 已有结果如何使用

自动读取已有 R25–R28（composition）和 R33–R36（structure）结果，只选与本轮相同的种子。历史结果目录保持不变，新比较表写到 `results\paper_revision`。

预检核对验证/测试集材料 ID、标签和训练配置；新配置另固定了监督数据和划分 CSV 的 SHA256。检测到这些文件被重新生成或更改时，任务会停下报错。新训练也保存代码、运行环境和图缓存文件状态的记录，以防不同版本结果被静默混用。图缓存检查为存在性、大小和修改时间，并非对每个图文件做二进制哈希。

显存不足时沿用项目原来的 batch size 回退机制，并记录实际 batch size；若出现回退，需要在正式方法对比中核对其与历史基线的差异。

## 结果在哪里看

首先看：

`D:\ck\two\results\paper_revision\summary_3seeds\seed_mean_std.csv`

然后看同目录的：

- `ensemble_comparison.csv`：验证集/测试集、原始/校准后的 AP、MCC、NLL、Brier、ECE，以及多种不确定性分数对应的风险指标。
- `member_comparison.csv`：逐种子结果，保留实际 batch size 和最佳 epoch。
- `paired_seed_differences.csv`：同划分、同种子下相对 Full 的 Joint 指标差值。
- 每个实验子目录的 `calibration_and_logic.json`：温度、验证集所选阈值、成员和集成层面的逻辑违反率。
- 每个实验子目录的 `test_reliability.csv`、`test_risk_coverage.csv`：用于重画论文校准图、风险覆盖图的数据。风险曲线文件最多保留 201 个绘图点；AURC 仍使用全部样本的覆盖水平计算。

AP 使用 `average_precision_score`，种子间标准差使用 `ddof=1`。不同方法的成员数和种子固定一致。这里的三种子差值和均值/标准差不能代替有足够统计功效的显著性证据；不要根据测试集最好的种子挑结果。

运行记录：`D:\ck\two\results\paper_revision\_runner\status.json`；同目录保存日志和 `preflight.json`。

## 其余任务

| 任务 | 什么时候用 |
|---|---|
| `59 - Paper revision - preflight only` | 只检查，不训练；60 已经包含这一步 |
| `61 - Paper revision - extend to 30 runs (5 seeds)` | 增加 2028、2029；若 18 次已完成，只新增 12 次训练；自动产生 `summary_5seeds` |
| `62 - Paper revision - summarize 3 seeds` | 18 次全部完成后，单独重新汇总三种子结果 |
| `63 - Paper revision - summarize 5 seeds` | 30 次全部完成后，单独重新汇总五种子结果 |

建议正式投稿时统一报告五个种子；是否扩到五种子应按事先确定的计算预算决定，不以三种子测试结果是否“有利”决定。

也可在 VS Code 的 PowerShell 终端直接运行：

```powershell
Set-Location -LiteralPath 'D:\ck\two'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_paper_revision.ps1 -Mode pilot
```

## 与已发表 DFT 外部验证的关系

本次补的是论文前两节的模型证据；这 18/30 次是内部回顾性补充实验。**它们不等于第三节的外部 DFT 验证已经完成。**

已发表 DFT 的材料名单、结构与证据需要另行整理，并完成与训练/验证数据的结构及组成排重。只有动力学证据的材料不能冒充具有完整 T、D、J 标签的外部样本。最终外部集合冻结之后，若必须剔除已进入训练的数据，还需重新训练匹配的模型与基线；这些训练不包含在上述 18/30 次中。

本次没有启动新的 DFT，也没有把未核验的文献材料自动写成外部真值。
