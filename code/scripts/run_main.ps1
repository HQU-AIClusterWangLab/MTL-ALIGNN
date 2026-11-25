. "$PSScriptRoot\common.ps1"

$tabular = @(
    "train/experiments/main/M01_random_forest.yaml",
    "train/experiments/main/M02_xgboost.yaml"
)
foreach ($experiment in $tabular) {
    Invoke-TwoPython -Label "Train $experiment" -Arguments @(
        "train/train_tabular.py", "--experiment", $experiment
    )
    $name = [System.IO.Path]::GetFileNameWithoutExtension($experiment)
    Invoke-TwoPython -Label "Summarize $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir", "D:/ck/two/results/main/$name"
    )
}

$gnn = @(
    "train/experiments/main/M03_cgcnn.yaml",
    "train/experiments/main/M04_single_alignn.yaml",
    "train/experiments/main/M05_equal_mtl.yaml",
    "train/experiments/main/M06_full_mtl.yaml"
)
foreach ($experiment in $gnn) {
    Invoke-TwoPython -Label "Train $experiment" -Arguments @(
        "train/train_gnn.py", "--experiment", $experiment
    )
    $name = [System.IO.Path]::GetFileNameWithoutExtension($experiment)
    Invoke-TwoPython -Label "Summarize $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir", "D:/ck/two/results/main/$name"
    )
}

$full = "D:/ck/two/results/main/M06_full_mtl"
$single = "D:/ck/two/results/main/M04_single_alignn"
Invoke-TwoPython -Label "Aggregate five-member Deep Ensemble and calibrate" -Arguments @(
    "train/aggregate_ensemble.py", "--experiment-dir", $full,
    "--calibration-bins", "15"
)
Invoke-TwoPython -Label "Run MC Dropout (30 stochastic passes)" -Arguments @(
    "train/mc_dropout_predict.py", "--experiment-dir", $full,
    "--seed", "2025", "--samples", "30"
)
Invoke-TwoPython -Label "Evaluate calibration and uncertainty quality" -Arguments @(
    "train/evaluate_uncertainty.py", "--full-experiment-dir", $full,
    "--single-experiment-dir", $single, "--calibration-bins", "15"
)
Invoke-TwoPython -Label "Compare direct Joint and Thermo/Dynamic fusion" -Arguments @(
    "train/evaluate_joint_fusion.py", "--full-experiment-dir", $full
)
Invoke-TwoPython -Label "Evaluate probability and risk-aware ranking" -Arguments @(
    "train/evaluate_ranking.py", "--full-experiment-dir", $full
)
Invoke-TwoPython -Label "Analyze gradient conflict for equal-weight MTL" -Arguments @(
    "train/analyze_task_gradients.py",
    "--full-experiment-dir", "D:/ck/two/results/main/M05_equal_mtl",
    "--seed", "2025", "--max-batches", "100"
)
Invoke-TwoPython -Label "Analyze gradient conflict for full adaptive MTL" -Arguments @(
    "train/analyze_task_gradients.py", "--full-experiment-dir", $full,
    "--seed", "2025", "--max-batches", "100"
)
Invoke-TwoPython -Label "Export current paper tables and figures" -Arguments @(
    "train/export_paper_artifacts.py"
)
Write-Host "Main experiments completed successfully."
