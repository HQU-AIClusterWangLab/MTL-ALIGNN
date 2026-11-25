# Ir3BrI2Te3 DFT / phonon reproducibility workspace

This directory is a fail-closed preparation workspace for the review-requested
`1BrI2Ir3Te3-1` case. It contains **no computed total energy, force, phonon
frequency, or frozen-mode stability result**. The only source structure is the
rank-14 CIF produced by the full 8,803-sample Unknown-pool screen:

`results/main/M10_unknown_screening/probability_top100_cif/014_1BrI2Ir3Te3-1.cif`

Its SHA-256 is recorded in the generated manifest, so a later calculation cannot
silently use a different geometry.

## Current blockers (checked 2026-08-09)

- ASE 3.28.0 is available in the project Python environment.
- No `vasp_std` or `vasp_gam` executable is on `PATH`.
- No `pw.x` / `ph.x` Quantum ESPRESSO executable is on `PATH`.
- Neither the phonopy CLI nor the phonopy Python module is installed.
- No VASP `POTCAR` is present in this project. A licensed POTCAR must not be
  fabricated or redistributed.
- No explicit Quantum ESPRESSO UPF pseudopotential set has been supplied.

Consequently, this machine cannot currently produce defensible energy or phonon
results for the material. `input_manifest.json` always records all scientific
claims as false and `ready_to_run` as false.

## Prepare the immutable structure snapshot

From the project root:

```powershell
& D:\ck\anaconda_env\envs\two\python.exe `
  dft\ir3bri2te3_phonon\prepare_inputs.py
```

The command validates the exact `Br1 I2 Ir3 Te3` composition and the requested
vacuum diagnostic, groups species deterministically, and creates:

- `prepared/Ir3BrI2Te3_reference.cif`
- `prepared/POSCAR`
- `prepared/input_manifest.json`

It refuses to overwrite these files unless `--force` is supplied. The manifest
contains SHA-256 hashes for the source CIF and generated structures.

All review-sensitive parameters are command-line options. For example:

```powershell
& D:\ck\anaconda_env\envs\two\python.exe `
  dft\ir3bri2te3_phonon\prepare_inputs.py `
  --supercell 3 3 1 `
  --displacement-amplitude-angstrom 0.01 `
  --force-convergence-ev-per-angstrom 0.01 `
  --kpoints 9 9 1 `
  --minimum-vacuum-gap-angstrom 15 `
  --vdw "D3(BJ)" `
  --magnetism scan `
  --soc
```

The vacuum number is explicitly defined as the largest wrapped inter-plane gap
along the c-normal direction. It does not subtract atomic radii and therefore is
a reproducible geometry diagnostic, not a claim about surface-to-surface vacuum.

## Required production calculation sequence

1. Select one calculator and record its exact version. Supply a consistent,
   cited pseudopotential family for Br, I, Ir, and Te. For VASP, pass the assembled
   licensed file with `--potcar-path`; the script records its hash but never copies
   it. For QE, extend the manifest with the exact UPF filenames and hashes before
   running.
2. Converge plane-wave cutoff, in-plane k mesh, and vacuum independently. The
   defaults (520 eV, 9x9x1, 15 A gap) are starting values, not convergence proof.
3. Relax the structure to the recorded force tolerance while keeping the 2D
   boundary treatment explicit. Compare the stated vdW correction and magnetic
   variants; because Ir and I are heavy, repeat the final electronic/force
   validation with the recorded SOC choice. The default magnetic scan deliberately
   has no invented starting moments; provide reviewed values explicitly, for
   example with repeated `--initial-moment Ir=...` arguments.
4. With the converged settings, use phonopy finite displacements with the recorded
   supercell and amplitude. Calculate every displaced-supercell force using the
   same functional, pseudopotentials, cutoff, k-point density, magnetism, vdW, and
   SOC policy. Archive the displaced structures, force files, `FORCE_SETS`, phonopy
   configuration, band path, and checksums.
5. Report acoustic-sum-rule and imaginary-frequency tolerances, and verify any
   soft mode against denser settings and a larger supercell before interpreting it.

## Frozen-mode rule

Frozen-mode geometries must not be generated from a frequency alone. They require
the exact q-point, branch index, complex eigenvector, atom order, units, and
normalization convention from the converged phonon calculation. None exists in
this workspace, so the manifest marks frozen-mode preparation as
`BLOCKED_UNTIL_PHONON_EIGENVECTOR_EXISTS`. This fail-closed state prevents an
invented distortion or a false stability conclusion.
