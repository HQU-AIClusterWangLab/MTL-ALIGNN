from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

from data.utils import load_config, write_json


def main() -> None:
    cfg = load_config()
    expected_python = Path(cfg["project"]["python"]).resolve()
    actual_python = Path(sys.executable).resolve()
    report = {
        "expected_python": str(expected_python),
        "actual_python": str(actual_python),
        "python_matches": expected_python == actual_python,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }

    import torch
    import dgl
    import alignn
    import ase
    import pandas
    import sklearn

    report.update(
        {
            "torch": torch.__version__,
            "torch_cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "dgl": dgl.__version__,
            "alignn": getattr(alignn, "__version__", "unknown"),
            "ase": ase.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
        }
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the configured environment.")
    device = torch.device("cuda:0")
    x = torch.randn(512, 512, device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "name": props.name,
        "memory_gb": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "smoke_mean": float(y.mean().cpu()),
    }

    db_path = Path(cfg["paths"]["db_path"])
    report["db_exists"] = db_path.exists()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    output = Path(cfg["paths"]["processed_dir"]) / "environment_report.json"
    write_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["python_matches"]:
        raise RuntimeError(
            "Wrong Python interpreter. Use "
            f"{cfg['project']['python']} instead of {sys.executable}."
        )


if __name__ == "__main__":
    main()
