$ErrorActionPreference = "Stop"
$ProjectRoot = "D:\ck\two"
$Python = "D:\ck\anaconda_env\envs\two\python.exe"

if (-not (Test-Path $Python)) {
    throw "Configured Python interpreter does not exist: $Python"
}
if (-not (Test-Path $ProjectRoot)) {
    throw "Project root does not exist: $ProjectRoot"
}

Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"

function Invoke-TwoPython {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host ""
    Write-Host ("=" * 88)
    Write-Host $Label
    Write-Host "$Python $($Arguments -join ' ')"
    Write-Host ("=" * 88)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-TwoScript {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )
    Write-Host ""
    Write-Host ("#" * 88)
    Write-Host $Label
    Write-Host ("#" * 88)
    & $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}
