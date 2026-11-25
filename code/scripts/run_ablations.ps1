. "$PSScriptRoot\common.ps1"
$experiments = @(
    "train/experiments/ablation/A01_joint_thermo.yaml",
    "train/experiments/ablation/A02_joint_dynamic.yaml",
    "train/experiments/ablation/A03_no_logic.yaml",
    "train/experiments/ablation/A04_fixed_weight.yaml",
    "train/experiments/ablation/A05_shuffle_auxiliary.yaml",
    "train/experiments/ablation/A06_adaptive_all.yaml"
)
foreach ($experiment in $experiments) {
    Invoke-TwoPython -Label "Train ablation $experiment" -Arguments @(
        "train/train_gnn.py", "--experiment", $experiment
    )
    $name = [System.IO.Path]::GetFileNameWithoutExtension($experiment)
    Invoke-TwoPython -Label "Summarize $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir", "D:/ck/two/results/ablation/$name"
    )
}
Invoke-TwoPython -Label "Refresh paper artifacts after ablations" -Arguments @(
    "train/export_paper_artifacts.py"
)
Write-Host "Ablation experiments completed successfully."
