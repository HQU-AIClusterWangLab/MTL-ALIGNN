param(
    [string]$Python = "D:\ck\anaconda_env\envs\two\python.exe",
    [int]$BootstrapReplicates = 2000,
    [int]$BootstrapSeed = 20260809
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    & $Python -m train.run_review_supplements `
        --project-root $ProjectRoot `
        --bootstrap-replicates $BootstrapReplicates `
        --bootstrap-seed $BootstrapSeed
    if ($LASTEXITCODE -ne 0) {
        throw "Review supplement analysis failed with exit code $LASTEXITCODE"
    }

    & $Python -m train.plot_review_supplements `
        --input-dir "$ProjectRoot\results\review_supplements"
    if ($LASTEXITCODE -ne 0) {
        throw "Review supplement figure generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
