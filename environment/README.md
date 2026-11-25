# Runtime environment

`environment_expected.txt` is the environment record supplied with the source project. `installed_versions.json` records the Python and package versions queried from the existing `D:\ck\anaconda_env\envs\two\python.exe` environment during organization. It is not a complete dependency lockfile.

Local examples:

```powershell
cd D:\ck\MTL-ALIGNN
& 'D:\ck\anaconda_env\envs\two\python.exe' predict.py --dry-run
& 'D:\ck\anaconda_env\envs\two\python.exe' train_fold.py --dry-run
```

No dependencies were installed or upgraded during organization. Full training, GPU inference, and CASTEP calculations require their corresponding software environments. Organizing the files does not rerun all experiments. See the root README for explicit graph-cache regeneration before actual inference or GNN training.
