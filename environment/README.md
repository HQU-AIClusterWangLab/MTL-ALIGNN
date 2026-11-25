# 运行环境

`environment_expected.txt` 为源项目提供的环境记录。`installed_versions.json` 是整理时通过现有 `D:\ck\anaconda_env\envs\two\python.exe` 查询得到的 Python 和包版本；它不是完整的依赖锁文件。

本机运行示例：

```powershell
cd D:\ck\MTL-ALIGNN
& 'D:\ck\anaconda_env\envs\two\python.exe' predict.py --dry-run
& 'D:\ck\anaconda_env\envs\two\python.exe' train_fold.py --dry-run
```

整理过程没有安装或升级依赖。完整训练、GPU 推理和 CASTEP 计算需相应软件环境；目录整理不等于重新运行全部实验。
