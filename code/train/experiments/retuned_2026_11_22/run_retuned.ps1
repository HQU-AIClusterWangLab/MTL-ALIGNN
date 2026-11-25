param(
    [switch]$IncludeUnknown
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = "D:\ck\two"
$PythonExe = "D:\ck\anaconda_env\envs\two\python.exe"
$ConfigDir = Join-Path $ProjectRoot "train\experiments\retuned_2026_11_22"
$ResultRoot = Join-Path $ProjectRoot "results\retuned_2026_11_22"
$Names = @(
    "M04_single_alignn",
    "M05_equal_mtl",
    "A03_no_logic",
    "A04_fixed_weight",
    "M06_full_mtl"
)

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found: $PythonExe"
}

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code $LASTEXITCODE : $($Arguments -join ' ')"
    }
}

Set-Location -LiteralPath $ProjectRoot
Invoke-CheckedPython -Arguments @(
    "-m", "train.experiments.retuned_2026_11_22.check_retuned_configs"
)

foreach ($Name in $Names) {
    $Config = Join-Path $ConfigDir "$Name.yaml"
    $Result = Join-Path $ResultRoot $Name
    Write-Host "==> Training $Name with seeds 2025-2029"
    # The training entry point skips seeds with COMPLETED.json on a rerun.
    Invoke-CheckedPython -Arguments @(
        "-m", "train.train_gnn", "--experiment", $Config
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "train.summarize_experiment", "--experiment-dir", $Result
    )
}

$Single = Join-Path $ResultRoot "M04_single_alignn"
$Full = Join-Path $ResultRoot "M06_full_mtl"
Invoke-CheckedPython -Arguments @(
    "-m", "train.experiments.retuned_2026_11_22.aggregate_single_joint",
    "--experiment-dir", $Single
)
Invoke-CheckedPython -Arguments @(
    "-m", "train.aggregate_ensemble", "--experiment-dir", $Full,
    "--calibration-bins", "15"
)

if ($IncludeUnknown) {
    # Dropout is zero in this configuration, so the ranking uses the
    # pre-specified five-member disagreement score instead of MC dropout.
    $SelectionDir = Join-Path $Full "uncertainty_evaluation"
    New-Item -ItemType Directory -Path $SelectionDir -Force | Out-Null
    $Selection = @{
        method = "deep_ensemble"
        score = "predictive_std"
        column = "p_joint_std"
        selection_set = "pre_specified"
        selection_rule = "five raw-member sample standard deviation"
    } | ConvertTo-Json -Depth 3
    [System.IO.File]::WriteAllText(
        (Join-Path $SelectionDir "uncertainty_selection.json"),
        $Selection,
        (New-Object System.Text.UTF8Encoding($false))
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "train.evaluate_ranking", "--full-experiment-dir", $Full
    )
    Invoke-CheckedPython -Arguments @(
        "-m", "train.screen_unknown", "--full-experiment-dir", $Full,
        "--output-dir", (Join-Path $ResultRoot "M10_unknown_screening")
    )
}

Write-Host "Completed. New results: $ResultRoot"
