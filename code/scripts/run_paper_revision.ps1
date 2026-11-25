param(
    [ValidateSet('preflight', 'pilot', 'full5', 'summary3', 'summary5')]
    [string]$Mode = 'pilot',
    [string]$Python = 'D:\ck\anaconda_env\envs\two\python.exe',
    [string]$ResultsDir = '',
    [switch]$AllowCpu
)
$ErrorActionPreference = 'Stop'
$revisionRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python was not found: $Python"
}
$previousPythonPath = $env:PYTHONPATH
$previousPythonUtf8 = $env:PYTHONUTF8
Push-Location -LiteralPath $revisionRoot
try {
    $env:PYTHONPATH = $revisionRoot
    $env:PYTHONUTF8 = '1'
    $revisionArguments = @('-u', '-m', 'train.run_paper_revision', '--mode', $Mode)
    if ($ResultsDir) { $revisionArguments += @('--results-dir', $ResultsDir) }
    if ($AllowCpu) { $revisionArguments += '--allow-cpu' }
    & $Python @revisionArguments
    if ($LASTEXITCODE -ne 0) { throw "Paper revision task failed (exit $LASTEXITCODE). See results/paper_revision/_runner/status.json." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONUTF8 = $previousPythonUtf8
    Pop-Location
}
