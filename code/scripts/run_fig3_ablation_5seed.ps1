$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = "D:\ck\two"
$PythonExe = "D:\ck\anaconda_env\envs\two\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python interpreter not found: $PythonExe"
}

function Invoke-Figure3Python {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Set-Location -LiteralPath $ProjectRoot

# Four configurations used by the Figure 3 ablation table.
# Existing seeds are detected from COMPLETED.json and are never submitted again.
$seeds = @(2025, 2026, 2027, 2028, 2029)
$experiments = @(
    @{
        Label = "Multitask (equal weighting)"
        File = "train/experiments/main/M05_equal_mtl.yaml"
        ResultDir = "D:/ck/two/results/main/M05_equal_mtl"
    },
    @{
        Label = "Multitask + AW"
        File = "train/experiments/ablation/A03_no_logic.yaml"
        ResultDir = "D:/ck/two/results/ablation/A03_no_logic"
    },
    @{
        Label = "Multitask + LC"
        File = "train/experiments/ablation/A04_fixed_weight.yaml"
        ResultDir = "D:/ck/two/results/ablation/A04_fixed_weight"
    },
    @{
        Label = "Multitask + AW + LC"
        File = "train/experiments/main/M06_full_mtl.yaml"
        ResultDir = "D:/ck/two/results/main/M06_full_mtl"
    }
)

foreach ($experiment in $experiments) {
    $missingSeeds = @(
        foreach ($seed in $seeds) {
            $marker = Join-Path $experiment.ResultDir "seed_$seed/COMPLETED.json"
            if (-not (Test-Path -LiteralPath $marker)) {
                $seed
            }
        }
    )

    if ($missingSeeds.Count -eq 0) {
        Write-Host "[SKIP] $($experiment.Label): seeds 2025-2029 are complete."
        continue
    }

    Write-Host "[RUN ] $($experiment.Label): missing seeds $($missingSeeds -join ', ')."
    Invoke-Figure3Python -Label "Train $($experiment.Label)" -Arguments @(
        "-m", "train.train_gnn", "--experiment", $experiment.File
    )

    Invoke-Figure3Python -Label "Summarize $($experiment.Label)" -Arguments @(
        "-m", "train.summarize_experiment",
        "--experiment-dir", $experiment.ResultDir
    )
}

Write-Host "Figure 3 five-seed ablation completion pass finished."
