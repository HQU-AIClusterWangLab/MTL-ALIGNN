param(
    [switch]$Force
)

. "$PSScriptRoot\common.ps1"
$full = "D:/ck/two/results/main/M06_full_mtl"

Invoke-TwoPython -Label "Validate/repair graphs for all Unknown materials" -Arguments @(
    "data/build_graphs.py",
    "--subset", "unknown",
    "--verify-existing"
)

$screenArguments = @(
    "train/screen_unknown.py",
    "--full-experiment-dir", $full
)
if ($Force) { $screenArguments += "--force" }
Invoke-TwoPython -Label "Screen Unknown pool and export Top-100/CIF" -Arguments $screenArguments

Invoke-TwoPython -Label "Export final paper tables and figures" -Arguments @(
    "train/export_paper_artifacts.py"
)

Write-Host "Unknown screening completed successfully."
