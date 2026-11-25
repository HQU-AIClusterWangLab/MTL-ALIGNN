from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ase.db import connect

from data.utils import load_config, row_to_mapping, write_json


def main() -> None:
    cfg = load_config()
    db = connect(cfg["paths"]["db_path"])
    fields = cfg["database"]
    required = [
        fields["id_field"],
        fields["ehull_field"],
        fields["dynamic_field"],
    ]
    key_counts: Counter[str] = Counter()
    dynamic_counts: Counter[str] = Counter()
    missing = Counter()
    sample = []

    for index, row in enumerate(db.select()):
        mapping = row_to_mapping(row)
        key_counts.update(mapping.keys())
        for field in required:
            if field not in mapping or mapping[field] is None:
                missing[field] += 1
        dynamic_counts[str(mapping.get(fields["dynamic_field"], "")).strip()] += 1
        if index < 10:
            sample.append(
                {
                    "row_id": row.id,
                    "uid": mapping.get(fields["id_field"]),
                    "formula": mapping.get(fields["formula_field"], row.formula),
                    "ehull": mapping.get(fields["ehull_field"]),
                    "dyn_stab": mapping.get(fields["dynamic_field"]),
                }
            )

    report = {
        "row_count": db.count(),
        "required_fields": required,
        "missing_required": dict(missing),
        "dynamic_counts": dict(dynamic_counts),
        "most_common_keys": key_counts.most_common(100),
        "sample": sample,
    }
    output = Path(cfg["paths"]["processed_dir"]) / "database_report.json"
    write_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise RuntimeError(f"Required database fields are missing: {dict(missing)}")


if __name__ == "__main__":
    main()
