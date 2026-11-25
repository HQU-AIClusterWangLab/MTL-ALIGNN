import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from ase.io import read


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "dft" / "ir3bri2te3_phonon" / "prepare_inputs.py"
SPEC = importlib.util.spec_from_file_location("ir3bri2te3_prepare_inputs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class Ir3BrI2Te3PreparationTests(unittest.TestCase):
    def _arguments(self, output_dir: Path) -> argparse.Namespace:
        return PREPARE.build_parser().parse_args(["--output-dir", str(output_dir)])

    def test_preparation_writes_valid_structure_and_fail_closed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "prepared"
            manifest = PREPARE.prepare_inputs(self._arguments(output_dir))

            self.assertEqual(manifest["status"], "INPUTS_PREPARED_NOT_CALCULATED")
            self.assertFalse(manifest["readiness"]["ready_to_run"])
            self.assertFalse(manifest["claims"]["dft_energy_complete"])
            self.assertFalse(manifest["claims"]["phonon_calculation_complete"])
            self.assertEqual(manifest["material"]["formula"], "BrI2Ir3Te3")
            self.assertEqual(manifest["material"]["atom_count"], 9)
            self.assertEqual(
                manifest["protocol"]["finite_displacement_supercell"], [3, 3, 1]
            )
            self.assertEqual(manifest["protocol"]["kpoint_mesh"], [9, 9, 1])
            self.assertGreaterEqual(
                manifest["structure"]["largest_interplane_gap_angstrom"],
                15.0 - 1.0e-8,
            )

            for filename in PREPARE.OUTPUT_FILENAMES:
                self.assertTrue((output_dir / filename).is_file())

            prepared = read(output_dir / "POSCAR")
            self.assertEqual(prepared.get_chemical_formula(mode="hill"), "BrI2Ir3Te3")
            on_disk = json.loads(
                (output_dir / "input_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                on_disk["material"]["source_cif_sha256"],
                manifest["material"]["source_cif_sha256"],
            )

    def test_existing_outputs_are_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "prepared"
            arguments = self._arguments(output_dir)
            PREPARE.prepare_inputs(arguments)
            with self.assertRaises(FileExistsError):
                PREPARE.prepare_inputs(arguments)

    def test_vacuum_requirement_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "prepared"
            arguments = self._arguments(output_dir)
            arguments.minimum_vacuum_gap_angstrom = 15.1
            with self.assertRaisesRegex(ValueError, "vacuum diagnostic"):
                PREPARE.prepare_inputs(arguments)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
