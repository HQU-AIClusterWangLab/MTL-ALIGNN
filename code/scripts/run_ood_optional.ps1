. "$PSScriptRoot\common.ps1"
Invoke-TwoPython -Label "Generate all OOD split definitions" -Arguments @(
    "data/make_ood_splits.py", "--mode", "all"
)
$experiments = @(
    "train/experiments/optional/O01_chemical_system_ood.yaml",
    "train/experiments/optional/O02_structure_family_ood.yaml"
)
foreach ($experiment in $experiments) {
    Invoke-TwoPython -Label "Train optional OOD experiment $experiment" -Arguments @(
        "train/train_gnn.py", "--experiment", $experiment
    )
    $name = [System.IO.Path]::GetFileNameWithoutExtension($experiment)
    Invoke-TwoPython -Label "Summarize $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir", "D:/ck/two/results/optional/$name"
    )
}
Invoke-TwoPython -Label "Compare IID and OOD performance" -Arguments @(
    "train/evaluate_ood.py",
    "--iid-dir", "D:/ck/two/results/main/M06_full_mtl",
    "--ood-dir", "D:/ck/two/results/optional/O01_chemical_system_ood",
    "--ood-dir", "D:/ck/two/results/optional/O02_structure_family_ood",
    "--output-dir", "D:/ck/two/results/optional/OOD_comparison"
)
Write-Host "Optional OOD experiments completed successfully."
