from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect

from data.utils import (
    anonymous_formula,
    chemical_system,
    composition_feature_dict,
    ensure_dir,
    finite_float,
    load_config,
    normalize_dynamic,
    row_to_mapping,
    structure_hash,
    write_json,
)


def cell_features(atoms) -> dict[str, float]:
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    volume = float(atoms.get_volume())
    a_vec, b_vec = np.asarray(atoms.cell.array[0]), np.asarray(atoms.cell.array[1])
    in_plane_area = float(np.linalg.norm(np.cross(a_vec, b_vec)))
    masses = np.asarray(atoms.get_masses(), dtype=float)
    density_proxy = float(masses.sum() / max(volume, 1e-12))
    areal_mass_proxy = float(masses.sum() / max(in_plane_area, 1e-12))
    return {
        "natoms": float(len(atoms)),
        "cell_a": float(lengths[0]),
        "cell_b": float(lengths[1]),
        "cell_c": float(lengths[2]),
        "cell_alpha": float(angles[0]),
        "cell_beta": float(angles[1]),
        "cell_gamma": float(angles[2]),
        "cell_volume": volume,
        "in_plane_area": in_plane_area,
        "mass_per_volume_proxy": density_proxy,
        "mass_per_area_proxy": areal_mass_proxy,
    }


def main() -> None:
    cfg = load_config()
    processed = ensure_dir(cfg["paths"]["processed_dir"])
    graph_dir = ensure_dir(cfg["paths"]["graph_dir"])
    db = connect(cfg["paths"]["db_path"])
    fields = cfg["database"]
    threshold = float(cfg["labels"]["ehull_threshold_eV_per_atom"])

    records: list[dict] = []
    failures: list[dict] = []
    dynamic_counts: Counter[str] = Counter()

    for row in db.select():
        mapping = row_to_mapping(row)
        try:
            atoms = row.toatoms()
            material_id = str(mapping[fields["id_field"]])
            ehull = finite_float(mapping[fields["ehull_field"]])
            if not np.isfinite(ehull):
                raise ValueError("ehull is missing or non-finite")
            dynamic_label, dynamic_raw = normalize_dynamic(
                mapping[fields["dynamic_field"]], cfg
            )
            dynamic_counts[dynamic_raw] += 1
            thermo_label = int(ehull <= threshold + 1e-12)
            joint_label = (
                int(thermo_label == 1 and dynamic_label == 1)
                if dynamic_label in (0, 1)
                else -1
            )

            formula = str(
                mapping.get(fields["formula_field"], atoms.get_chemical_formula())
            )
            # Use ASE's structure-derived reduced formula so equivalent formula
            # strings are grouped consistently during leakage-controlled splitting.
            reduced_formula = atoms.get_chemical_formula(mode="reduce")
            structure_digest = structure_hash(atoms)
            layergroup = mapping.get(fields["layergroup_field"])
            anon = anonymous_formula(formula)
            record = {
                "row_id": int(row.id),
                "material_id": material_id,
                "unique_id": str(mapping.get("unique_id", "")),
                "formula": formula,
                "reduced_formula": reduced_formula,
                "anonymous_formula": anon,
                "chemical_system": chemical_system(atoms),
                "structure_hash": structure_digest,
                "structure_family": f"{anon}|{layergroup}",
                "ehull": ehull,
                "dyn_stab_raw": dynamic_raw,
                "thermo_label": thermo_label,
                "dynamic_label": dynamic_label,
                "joint_label": joint_label,
                "screening_eligible": int(
                    dynamic_label == -1 and thermo_label == 1
                ),
                "spacegroup_symbol": mapping.get(
                    fields["spacegroup_symbol_field"]
                ),
                "spacegroup_number": mapping.get(
                    fields["spacegroup_number_field"]
                ),
                "layergroup": layergroup,
                "layergroup_number": mapping.get(
                    fields["layergroup_number_field"]
                ),
                "bravais_type": mapping.get(fields["bravais_field"]),
                "source_label": mapping.get(fields["source_label_field"]),
                "thickness": finite_float(
                    mapping.get(fields["thickness_field"])
                ),
                "graph_path": str(graph_dir / f"row_{row.id}.bin"),
            }
            record.update(cell_features(atoms))
            record.update(composition_feature_dict(atoms))
            records.append(record)
        except Exception as exc:
            failures.append(
                {
                    "row_id": getattr(row, "id", None),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    frame = pd.DataFrame(records).sort_values("row_id").reset_index(drop=True)
    if frame["material_id"].duplicated().any():
        duplicates = frame[frame["material_id"].duplicated(keep=False)].copy()
        duplicates.to_csv(processed / "duplicate_material_ids.csv", index=False)
        raise RuntimeError("The configured material_id field is not unique.")

    exact_duplicates = frame[
        frame["structure_hash"].duplicated(keep=False)
    ].sort_values(["structure_hash", "row_id"])
    exact_duplicates.to_csv(processed / "exact_structure_duplicates.csv", index=False)

    # Resolve exact-coordinate duplicates conservatively. Conflicting known
    # labels are excluded rather than resolved by database order. When one row
    # is labeled and another is Unknown, prefer the labeled row so the same
    # structure cannot reappear in the screening pool.
    selected_indices: list[int] = []
    conflicts: list[pd.DataFrame] = []
    resolution_rows: list[dict] = []
    for structure_digest, group in frame.groupby("structure_hash", sort=False):
        group = group.sort_values("row_id")
        known_dynamic = sorted(
            set(group.loc[group["dynamic_label"].isin([0, 1]), "dynamic_label"])
        )
        thermo_values = sorted(set(group["thermo_label"]))
        conflict_reason = None
        if len(known_dynamic) > 1:
            conflict_reason = "conflicting_dynamic_labels"
        elif len(thermo_values) > 1:
            conflict_reason = "conflicting_thermo_labels"

        if conflict_reason is not None:
            conflict = group.copy()
            conflict["conflict_reason"] = conflict_reason
            conflicts.append(conflict)
            resolution_rows.append(
                {
                    "structure_hash": structure_digest,
                    "group_size": int(len(group)),
                    "status": "excluded",
                    "reason": conflict_reason,
                    "selected_row_id": None,
                }
            )
            continue

        labeled = group[group["dynamic_label"].isin([0, 1])]
        representative = (labeled if not labeled.empty else group).iloc[0]
        selected_indices.append(int(representative.name))
        resolution_rows.append(
            {
                "structure_hash": structure_digest,
                "group_size": int(len(group)),
                "status": "kept",
                "reason": "prefer_known_then_lowest_row_id",
                "selected_row_id": int(representative["row_id"]),
            }
        )

    conflict_frame = (
        pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame()
    )
    conflict_frame.to_csv(
        processed / "conflicting_exact_structure_duplicates.csv", index=False
    )
    pd.DataFrame(resolution_rows).to_csv(
        processed / "duplicate_resolution.csv", index=False
    )
    deduplicated = frame.loc[selected_indices].sort_values("row_id").reset_index(drop=True)

    supervised = deduplicated[
        deduplicated["dynamic_label"].isin([0, 1])
    ].copy()
    unknown = deduplicated[
        deduplicated["dynamic_label"] == -1
    ].copy()
    eligible = unknown[unknown["thermo_label"] == 1].copy()

    frame.to_csv(processed / "all_metadata_before_dedup.csv", index=False)
    deduplicated.to_csv(processed / "all_metadata.csv", index=False)
    supervised.to_csv(processed / "supervised.csv", index=False)
    unknown.to_csv(processed / "unknown_pool.csv", index=False)
    eligible.to_csv(processed / "unknown_thermo_eligible.csv", index=False)
    pd.DataFrame(failures).to_csv(processed / "prepare_failures.csv", index=False)

    non_feature_columns = {
        "row_id", "material_id", "unique_id", "formula", "reduced_formula",
        "anonymous_formula", "chemical_system", "structure_hash",
        "structure_family", "ehull", "dyn_stab_raw", "thermo_label",
        "dynamic_label", "joint_label", "screening_eligible",
        "spacegroup_symbol", "layergroup", "bravais_type", "source_label",
        "graph_path",
    }
    # Exclude vacuum-sensitive 3D cell quantities from the traditional 2D
    # baseline. They depend on the arbitrary out-of-plane vacuum convention.
    non_feature_columns.update(
        {"cell_c", "cell_volume", "mass_per_volume_proxy", "cell_alpha", "cell_beta"}
    )
    tabular_features = [
        column
        for column in supervised.columns
        if column not in non_feature_columns
        and pd.api.types.is_numeric_dtype(supervised[column])
    ]
    # Explicitly exclude all target-defining fields.
    forbidden = {"ehull", "dynamic_label", "thermo_label", "joint_label"}
    if forbidden.intersection(tabular_features):
        raise RuntimeError("Target leakage detected in tabular features.")
    write_json(tabular_features, processed / "tabular_feature_columns.json")

    quadrant = (
        supervised.groupby(["thermo_label", "dynamic_label"], dropna=False)
        .agg(
            n=("material_id", "size"),
            ehull_mean=("ehull", "mean"),
            ehull_median=("ehull", "median"),
        )
        .reset_index()
    )
    quadrant["quadrant"] = quadrant.apply(
        lambda row: f"T{int(row.thermo_label):+d}/D{int(row.dynamic_label):+d}",
        axis=1,
    )
    quadrant.to_csv(processed / "thermo_dynamic_quadrants.csv", index=False)

    summary = {
        "database_rows": int(db.count()),
        "successfully_parsed_rows": int(len(frame)),
        "prepare_failures": int(len(failures)),
        "exact_duplicate_rows_reported": int(len(exact_duplicates)),
        "conflicting_duplicate_rows_excluded": int(len(conflict_frame)),
        "rows_after_exact_deduplication": int(len(deduplicated)),
        "supervised_rows": int(len(supervised)),
        "unknown_rows": int(len(unknown)),
        "unknown_thermo_eligible_rows": int(len(eligible)),
        "thermo_positive_supervised": int(supervised["thermo_label"].sum()),
        "dynamic_positive_supervised": int(supervised["dynamic_label"].sum()),
        "joint_positive_supervised": int(supervised["joint_label"].sum()),
        "dynamic_raw_counts": dict(dynamic_counts),
        "ehull_threshold_eV_per_atom": threshold,
        "tabular_feature_count": len(tabular_features),
    }
    write_json(summary, processed / "data_summary.json")
    print(summary)
    if failures:
        raise RuntimeError(
            f"{len(failures)} rows failed during preparation. "
            "Inspect data/processed/prepare_failures.csv."
        )


if __name__ == "__main__":
    main()
