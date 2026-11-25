param(
    [ValidateSet('preflight', 'analyze', 'full')]
    [string]$Mode = 'preflight'
)
$ErrorActionPreference = 'Stop'
$coreRoot = Split-Path -Parent $PSScriptRoot
$corePython = 'D:\ck\anaconda_env\envs\two\python.exe'
if (-not (Test-Path -LiteralPath $corePython -PathType Leaf)) { throw "Python not found: $corePython" }
$corePrevious = @{}
foreach ($key in @('PYTHONPATH', 'PYTHONUTF8', 'FIRST_SCI_DATA_DIR', 'PYTHONDONTWRITEBYTECODE')) {
    $corePrevious[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
Push-Location -LiteralPath $coreRoot
try {
    $env:PYTHONPATH = "$coreRoot\src"
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    if (-not $env:FIRST_SCI_DATA_DIR) { $env:FIRST_SCI_DATA_DIR = 'D:\ck\First sci\data' }
    # -m includes the project cwd, preserving the actual train/ package layout.
    & $corePython -B -u -m train.run_core_revision --mode $Mode --project-root $coreRoot
    if ($LASTEXITCODE -ne 0) { throw "Core revision failed (exit $LASTEXITCODE); see results/core_revision/_runner/status.json" }
} finally {
    foreach ($key in $corePrevious.Keys) { [Environment]::SetEnvironmentVariable($key, $corePrevious[$key], 'Process') }
    Pop-Location
}
