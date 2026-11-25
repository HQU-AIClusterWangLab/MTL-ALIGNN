# 实机诊断摘要

- 数据库路径：`D:\ck\two\data\c2db.db`
- 数据库记录数：16,905
- `dyn_stab`：
  - Unknown：8,803
  - Yes：4,791
  - No：3,311
- `ehull`完整记录：16,905
- `ehull <= 0.10 eV/atom`：5,688
- 结构读取抽查：10/10成功
- CUDA矩阵运算测试：成功
- GPU：RTX 4090 D，23.99 GB
- ALIGNN 2026.4.2接口：
  - `ALIGNNConfig(name="alignn", ...)`
  - `ALIGNN.forward((g, lg, lattice))`
  - `Graph.atom_dgl_multigraph(...)`
