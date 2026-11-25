#!/usr/bin/env python
"""Prepare an auditable Ir3BrI2Te3 DFT/phonon input snapshot.

This program performs structure conversion and input bookkeeping only. It does
not run a DFT code, generate forces, diagonalize a dynamical matrix, or claim a
phonon result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import ase
from ase import Atoms
from ase.io import read, write


MATERIAL_ID = "1BrI2Ir3Te3-1"
EXPECTED_COMPOSITION = {"Br": 1, "I": 2, "Ir": 3, "Te": 3}
OUTPUT_FILENAMES = (
    "Ir3BrI2Te3_reference.cif",
    "POSCAR",
    "input_manifest.json",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_cif() -> Path:
    return (
        project_root()
        / "results"
        / "main"
        / "M10_unknown_screening"
        / "probability_top100_cif"
        / f"014_{MATERIAL_ID}.cif"
    )


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "prepared"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _element_moments(entries: Iterable[str]) -> dict[str, float]:
    moments: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"invalid initial moment {entry!r}; expected an entry such as Ir=1.0"
            )
        element, raw_value = entry.split("=", 1)
        element = element.strip()
        if element not in EXPECTED_COMPOSITION:
            raise ValueError(
                f"initial moment element {element!r} is not in {sorted(EXPECTED_COMPOSITION)}"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"initial moment for {element} must be finite")
        moments[element] = value
    return moments


def _cell_height_and_largest_plane_gap(atoms: Atoms) -> tuple[float, float]:
    """Return the c-normal cell height and largest inter-plane gap in angstrom.

    The latter is a reproducible geometry diagnostic, not a van-der-Waals-surface
    vacuum thickness. Fractional z is appropriate here because the target is a
    2D slab whose non-periodic direction is the third lattice direction.
    """

    cell = atoms.cell.array.tolist()
    a, b = cell[0], cell[1]
    cross_ab = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    basal_area = math.sqrt(sum(component * component for component in cross_ab))
    if basal_area <= 0:
        raise ValueError("invalid cell: in-plane lattice vectors have zero area")
    height = abs(float(atoms.cell.volume)) / basal_area

    fractional_z = sorted(
        float(position[2]) for position in atoms.get_scaled_positions(wrap=True)
    )
    if not fractional_z:
        raise ValueError("reference structure contains no atoms")
    fractional_gaps = [
        fractional_z[index + 1] - fractional_z[index]
        for index in range(len(fractional_z) - 1)
    ]
    fractional_gaps.append(1.0 + fractional_z[0] - fractional_z[-1])
    return height, max(fractional_gaps) * height


def _validate_reference(atoms: Atoms, source: Path) -> None:
    counts = Counter(atoms.get_chemical_symbols())
    if dict(counts) != EXPECTED_COMPOSITION:
        raise ValueError(
            f"{source} has composition {dict(counts)}, expected {EXPECTED_COMPOSITION}"
        )
    if len(atoms) != sum(EXPECTED_COMPOSITION.values()):
        raise ValueError(f"{source} has an unexpected atom count: {len(atoms)}")
    if not all(bool(value) for value in atoms.pbc):
        raise ValueError("reference CIF must define a three-dimensional periodic cell")
    if abs(float(atoms.cell.volume)) <= 0:
        raise ValueError("reference CIF has a zero-volume cell")


def _species_grouped_structure(atoms: Atoms) -> tuple[Atoms, list[int]]:
    permutation = sorted(
        range(len(atoms)), key=lambda index: (atoms[index].symbol, index)
    )
    grouped = atoms[permutation]
    grouped.set_cell(atoms.cell)
    grouped.set_pbc(atoms.pbc)
    return grouped, permutation


def _executable_probe() -> dict[str, str | None]:
    return {
        executable: shutil.which(executable)
        for executable in ("vasp_std", "vasp_gam", "pw.x", "ph.x", "phonopy")
    }


def _potential_record(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "provided": False,
            "path": None,
            "sha256": None,
            "note": "No licensed VASP POTCAR was provided or copied.",
        }
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"--potcar-path is not a file: {resolved}")
    return {
        "provided": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "note": "The POTCAR is referenced and hashed, never copied by this script.",
    }


def _phonopy_module_available() -> bool:
    return importlib.util.find_spec("phonopy") is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cif", type=Path, default=default_source_cif())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--calculator", choices=("vasp", "qe"), default="vasp")
    parser.add_argument(
        "--supercell",
        type=_positive_int,
        nargs=3,
        metavar=("NA", "NB", "NC"),
        default=(3, 3, 1),
    )
    parser.add_argument(
        "--displacement-amplitude-angstrom", type=_positive_float, default=0.01
    )
    parser.add_argument(
        "--force-convergence-ev-per-angstrom", type=_positive_float, default=0.01
    )
    parser.add_argument(
        "--electronic-convergence-ev", type=_positive_float, default=1.0e-7
    )
    parser.add_argument("--plane-wave-cutoff-ev", type=_positive_float, default=520.0)
    parser.add_argument(
        "--kpoints",
        type=_positive_int,
        nargs=3,
        metavar=("NKX", "NKY", "NKZ"),
        default=(9, 9, 1),
    )
    parser.add_argument(
        "--minimum-vacuum-gap-angstrom",
        type=_positive_float,
        default=15.0,
        help=(
            "minimum largest inter-plane gap along c; this is a geometry diagnostic, "
            "not a radius-corrected surface vacuum"
        ),
    )
    parser.add_argument("--xc-functional", default="PBE")
    parser.add_argument("--vdw", default="D3(BJ)")
    parser.add_argument(
        "--magnetism",
        choices=("nonmagnetic", "collinear", "noncollinear", "scan"),
        default="scan",
        help="'scan' requires comparing nonmagnetic and initialized magnetic states",
    )
    parser.add_argument(
        "--initial-moment",
        action="append",
        default=[],
        metavar="ELEMENT=MU_B",
        help="repeatable element-level starting moment for a magnetic calculation",
    )
    parser.add_argument(
        "--soc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record whether spin-orbit coupling is required in the validation protocol",
    )
    parser.add_argument(
        "--potcar-path",
        type=Path,
        default=None,
        help="optional assembled POTCAR to hash and reference; it is never copied",
    )
    parser.add_argument("--force", action="store_true", help="overwrite only known outputs")
    return parser


def prepare_inputs(args: argparse.Namespace) -> dict[str, object]:
    source = args.source_cif.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source CIF does not exist: {source}")

    atoms = read(source, index=0)
    _validate_reference(atoms, source)
    grouped, permutation = _species_grouped_structure(atoms)
    cell_height, largest_gap = _cell_height_and_largest_plane_gap(grouped)
    if largest_gap + 1.0e-8 < args.minimum_vacuum_gap_angstrom:
        raise ValueError(
            "reference structure does not meet the requested vacuum diagnostic: "
            f"{largest_gap:.8f} < {args.minimum_vacuum_gap_angstrom:.8f} angstrom"
        )

    initial_moments = _element_moments(args.initial_moment)
    if args.magnetism == "nonmagnetic" and initial_moments:
        raise ValueError("initial moments cannot be used with --magnetism nonmagnetic")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing prepared files: {joined}")

    reference_cif = output_paths["Ir3BrI2Te3_reference.cif"]
    poscar = output_paths["POSCAR"]
    write(reference_cif, grouped, format="cif")
    write(poscar, grouped, format="vasp", direct=True, sort=False, vasp5=True)

    executables = _executable_probe()
    phonopy_module = _phonopy_module_available()
    potcar = _potential_record(args.potcar_path)
    vasp_available = bool(executables["vasp_std"] or executables["vasp_gam"])
    qe_available = bool(executables["pw.x"])
    calculator_available = vasp_available if args.calculator == "vasp" else qe_available
    potential_available = bool(potcar["provided"]) if args.calculator == "vasp" else False

    blockers: list[str] = []
    if not calculator_available:
        blockers.append(f"requested calculator executable is unavailable: {args.calculator}")
    if args.calculator == "vasp" and not potential_available:
        blockers.append("an explicit, licensed POTCAR was not provided")
    if args.calculator == "qe":
        blockers.append("explicit QE UPF pseudopotentials were not provided")
    if args.magnetism in {"collinear", "noncollinear", "scan"} and not initial_moments:
        blockers.append(
            "a magnetic calculation/scan was requested but no initial moments were supplied"
        )
    if not (executables["phonopy"] or phonopy_module):
        blockers.append("phonopy CLI and Python module are unavailable")
    blockers.append("no DFT energies or forces have been calculated")
    blockers.append("no phonon eigenvalues/eigenvectors have been calculated")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "INPUTS_PREPARED_NOT_CALCULATED",
        "claims": {
            "dft_energy_complete": False,
            "force_calculations_complete": False,
            "phonon_calculation_complete": False,
            "frozen_mode_calculation_complete": False,
        },
        "material": {
            "material_id": MATERIAL_ID,
            "formula": grouped.get_chemical_formula(mode="hill"),
            "atom_count": len(grouped),
            "composition": EXPECTED_COMPOSITION,
            "source_cif": str(source),
            "source_cif_sha256": sha256_file(source),
        },
        "structure": {
            "pbc": [bool(value) for value in grouped.pbc],
            "cell_angstrom": grouped.cell.array.tolist(),
            "species_order": grouped.get_chemical_symbols(),
            "grouped_atom_to_source_atom_index_zero_based": permutation,
            "c_normal_cell_height_angstrom": cell_height,
            "largest_interplane_gap_angstrom": largest_gap,
            "vacuum_diagnostic_definition": (
                "largest wrapped fractional-z gap multiplied by c-normal cell height; "
                "atomic radii are not subtracted"
            ),
        },
        "protocol": {
            "requested_calculator": args.calculator,
            "xc_functional": args.xc_functional,
            "vdw_correction": args.vdw,
            "plane_wave_cutoff_ev": args.plane_wave_cutoff_ev,
            "electronic_convergence_ev": args.electronic_convergence_ev,
            "ionic_force_convergence_ev_per_angstrom": args.force_convergence_ev_per_angstrom,
            "kpoint_mesh": list(args.kpoints),
            "minimum_vacuum_gap_angstrom": args.minimum_vacuum_gap_angstrom,
            "magnetism": args.magnetism,
            "initial_moments_mu_b_by_element": initial_moments,
            "spin_orbit_coupling_required": args.soc,
            "finite_displacement_supercell": list(args.supercell),
            "finite_displacement_amplitude_angstrom": args.displacement_amplitude_angstrom,
            "frozen_mode": {
                "status": "BLOCKED_UNTIL_PHONON_EIGENVECTOR_EXISTS",
                "required_inputs": [
                    "q-point",
                    "branch index",
                    "frequency with units",
                    "complex eigenvector with normalization convention",
                ],
            },
        },
        "environment_probe": {
            "python_version": sys.version.split()[0],
            "ase_version": ase.__version__,
            "executables": executables,
            "phonopy_python_module_available": phonopy_module,
            "vasp_potcar": potcar,
            "qe_pseudopotentials_provided": False,
        },
        "readiness": {
            "reference_structure_prepared": True,
            "requested_calculator_available": calculator_available,
            "pseudopotentials_explicitly_recorded": potential_available,
            "phonopy_available": bool(executables["phonopy"] or phonopy_module),
            "ready_to_run": False,
            "blocking_reasons": blockers,
        },
        "prepared_outputs": {
            "reference_cif": {
                "path": str(reference_cif),
                "sha256": sha256_file(reference_cif),
            },
            "poscar": {"path": str(poscar), "sha256": sha256_file(poscar)},
        },
    }

    manifest_path = output_paths["input_manifest.json"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = prepare_inputs(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(manifest["readiness"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
