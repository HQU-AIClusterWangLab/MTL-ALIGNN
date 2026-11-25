# 文献声子验证纳入协议

本接口只处理人工核实的真实资料，不自动生成 DOI、论文标题或声子结论。

1. 候选必须来自现有 `data/processed/unknown_pool.csv` 的 8803 个材料，使用唯一 material_id 关联。先按事先确定的候选池范围或模型排序确定待检索材料，再查文献；不能先找成功案例再反选材料。开始检索前在 notes 记录选材规则、检索日期、检索词及数据库；未找到适用文献也记录在独立检索日志中。
2. 化学式相同不等于结构相同。优先对应空间群、晶格参数、原子排布、层数、相/多型及计算维度。在 structure_match_basis 写明具体结构证据、文献页码/图表与偏差。
3. structure_match_status 仅允许 matched、partial、uncertain、excluded。matched 和 excluded 均保留审计记录；不得删除失败案例。资料不足用 uncertain，有部分对应用 partial，有明确不相符证据用 excluded。
4. literature_dynamic_conclusion 仅允许 stable、unstable、ambiguous。应检查声子计算设置、采样路径、虚频位置及作者解释；不能只按标题判断。结论不确定时填 ambiguous，不把没有声子证据写成 stable。
5. 文献 phonon 只直接验证 Dynamic。它不能自动证明本文 Joint-positive；Joint 还需要满足本文 Ehull ≤ 0.10 eV/atom 条件，并确保对应同一结构和适用计算条件。
6. DOI、标题、年份只能从真实来源人工填写；source_title/doi 至少提供一个可核查来源。notes 保留结构证据、动力学证据、计算条件及纳入/排除理由。多篇来源可对同一 material_id 分行记录，冲突结论不得择优删除。
7. 模板中的 formula/rank/probability 可留空。脚本合并真实模型输出后保留人工快照为 manual_*，模型列来自已有 unknown screening 输出。model_rank 按全部 8803 候选的 calibrated pJ 降序，平分按 material_id；u_dynamic/u_joint 明确使用已有 raw member std（ddof=1）。
8. Dynamic 一致性仅统计 matched 且 stable/unstable 的记录。partial、uncertain、excluded、ambiguous 不进入一致率分母，但全部保留。记录层与候选层分别输出；同一候选存在矛盾来源时标记冲突，不掩盖。
9. 默认 Dynamic 决策阈值来自生成 unknown predictions 的 M06 五成员集成 validation 校准文件。若更换预测文件，必须同时提供对应的冻结 thresholds.json，不能用文献标签或 test 标签重新选阈值。

在项目根目录运行：

```powershell
& 'D:\ck\anaconda_env\envs\two\python.exe' -B src/train/analyze_literature_validation.py
```

模板为空时只打印提示并正常退出，不创建科学结果。人工填写后结果写入 `results/core_revision/06_literature_validation/`；输入变化时保留旧版本，新增 `revisions/<signature>/`。
