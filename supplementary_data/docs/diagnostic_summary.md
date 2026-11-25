# Original-machine diagnostic summary

- Database path: `D:\ck\two\data\c2db.db`
- Database records: 16,905
- `dyn_stab`:
  - Unknown: 8,803
  - Yes: 4,791
  - No: 3,311
- Complete `ehull` records: 16,905
- `ehull <= 0.10 eV/atom`: 5,688
- Structure-reading spot check: 10/10 successful
- CUDA matrix-operation test: passed
- GPU: RTX 4090 D, 23.99 GB
- ALIGNN 2026.4.2 interface:
  - `ALIGNNConfig(name="alignn", ...)`
  - `ALIGNN.forward((g, lg, lattice))`
  - `Graph.atom_dgl_multigraph(...)`
