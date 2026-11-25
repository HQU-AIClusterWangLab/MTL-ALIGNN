from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from data.make_review_splits import (
    LABEL_COLUMNS,
    add_canonical_group_columns,
    assignment_sha256,
    audit_assignments,
    canonicalize_element_counts,
    make_group_disjoint_assignments,
    make_random_stratified_assignments,
)


def test_canonical_composition_is_order_and_supercell_invariant() -> None:
    primitive = canonicalize_element_counts([("S", 2), ("Mo", 1)])
    reordered_supercell = canonicalize_element_counts({"Mo": 4, "S": 8})

    assert primitive == reordered_supercell
    assert primitive.composition_id == "Mo=1|S=2"
    assert primitive.formula == "MoS2"
    assert primitive.chemical_system_id == "Mo-S"


def test_canonical_composition_uses_hill_order_and_rejects_bad_counts() -> None:
    composition = canonicalize_element_counts({"O": 6, "H": 12, "C": 6})
    assert composition.formula == "CH2O"
    assert composition.composition_id == "C=1|H=2|O=1"
    with unittest.TestCase().assertRaisesRegex(ValueError, "not integer-valued"):
        canonicalize_element_counts({"Mo": 1.5, "S": 2})
    with unittest.TestCase().assertRaisesRegex(ValueError, "At least one positive"):
        canonicalize_element_counts({"Mo": 0})


def _prepared_fixture(n_groups: int = 24, rows_per_group: int = 4) -> pd.DataFrame:
    rows = []
    row_id = 0
    for group_index in range(n_groups):
        for member in range(rows_per_group):
            row_id += 1
            thermo = (group_index + member) % 2
            dynamic = (group_index // 2 + member) % 2
            fractions = {f"frac_Z_{z:03d}": 0.0 for z in range(1, 119)}
            # Alternate MoS2 and doubled Mo2S4 cells to exercise count reduction.
            multiplier = 1 + (row_id % 2)
            natoms = 3 * multiplier
            fractions["frac_Z_016"] = 2.0 / 3.0
            fractions["frac_Z_042"] = 1.0 / 3.0
            rows.append(
                {
                    "row_id": row_id,
                    "material_id": f"m-{row_id:04d}",
                    "natoms": natoms,
                    "structure_family": f"family-{group_index:02d}",
                    "structure_hash": f"hash-{row_id:04d}",
                    "thermo_label": thermo,
                    "dynamic_label": dynamic,
                    "joint_label": thermo * dynamic,
                    **fractions,
                }
            )
    return pd.DataFrame(rows)


def test_fraction_columns_produce_one_supercell_invariant_composition() -> None:
    enriched = add_canonical_group_columns(_prepared_fixture(4, 4))
    assert enriched["canonical_reduced_composition_id"].nunique() == 1
    assert set(enriched["canonical_formula"]) == {"MoS2"}


def test_random_split_is_deterministic_complete_and_stratified() -> None:
    frame = _prepared_fixture()
    first = make_random_stratified_assignments(frame, seed=7)
    second = make_random_stratified_assignments(frame, seed=7)
    assert first.equals(second)
    assert first["material_id"].is_unique
    assert set(first["material_id"]) == set(frame["material_id"])
    assert set(first["split"]) == {"train", "val", "test"}
    merged = frame.merge(first, on="material_id", validate="one_to_one")
    overall = merged[list(LABEL_COLUMNS)].mean()
    for _, part in merged.groupby("split"):
        assert np.max(np.abs(part[list(LABEL_COLUMNS)].mean() - overall)) < 0.20
    assert assignment_sha256(first) == assignment_sha256(second.sample(frac=1))


def test_group_disjoint_split_and_audit_have_zero_target_overlap() -> None:
    frame = add_canonical_group_columns(_prepared_fixture())
    assignments = make_group_disjoint_assignments(
        frame,
        "structure_family_id",
        seed=11,
        trials=80,
    )
    audit = audit_assignments(
        frame,
        assignments,
        target_group_column="structure_family_id",
        enforce_target_disjoint=True,
    )
    assert audit["coverage"]["all_source_rows_covered_once"] is True
    assert audit["target_group_overlap_unique_count"] == 0
    assert audit["target_group_pairwise_overlap_count_sum"] == 0
    for pair in audit["pairwise_group_overlap"]["structure_family_id"].values():
        assert pair["overlap_group_count"] == 0


def test_audit_rejects_duplicate_or_incomplete_assignments() -> None:
    frame = add_canonical_group_columns(_prepared_fixture())
    assignments = make_random_stratified_assignments(frame, seed=3)
    duplicate = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    with unittest.TestCase().assertRaisesRegex(ValueError, "duplicate material_id"):
        audit_assignments(
            frame,
            duplicate,
            target_group_column=None,
            enforce_target_disjoint=False,
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "exactly cover"):
        audit_assignments(
            frame,
            assignments.iloc[:-1],
            target_group_column=None,
            enforce_target_disjoint=False,
        )


class ReviewSplitTests(unittest.TestCase):
    """Expose the functional assertions to the project's unittest runner."""

    def test_canonical_composition_is_order_and_supercell_invariant(self) -> None:
        test_canonical_composition_is_order_and_supercell_invariant()

    def test_canonical_composition_uses_hill_order_and_rejects_bad_counts(self) -> None:
        test_canonical_composition_uses_hill_order_and_rejects_bad_counts()

    def test_fraction_columns_produce_one_supercell_invariant_composition(self) -> None:
        test_fraction_columns_produce_one_supercell_invariant_composition()

    def test_random_split_is_deterministic_complete_and_stratified(self) -> None:
        test_random_split_is_deterministic_complete_and_stratified()

    def test_group_disjoint_split_and_audit_have_zero_target_overlap(self) -> None:
        test_group_disjoint_split_and_audit_have_zero_target_overlap()

    def test_audit_rejects_duplicate_or_incomplete_assignments(self) -> None:
        test_audit_rejects_duplicate_or_incomplete_assignments()
