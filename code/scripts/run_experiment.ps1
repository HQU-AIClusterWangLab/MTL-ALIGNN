param(
    [Parameter(Mandatory = $true)][string]$Experiment,
    [ValidateSet("gnn", "tabular")][string]$Kind = "gnn",
    [ValidateSet("main", "ablation", "optional", "review")][string]$Category = "main",
    [switch]$Force
)
. "$PSScriptRoot\common.ps1"
$entry = if ($Kind -eq "gnn") { "train/train_gnn.py" } else { "train/train_tabular.py" }
$arguments = @($entry, "--experiment", $Experiment)
if ($Force) { $arguments += "--force" }
Invoke-TwoPython -Label "Run experiment $Experiment" -Arguments $arguments
$name = [System.IO.Path]::GetFileNameWithoutExtension($Experiment)
$resultDir = "D:/ck/two/results/$Category/$name"
Invoke-TwoPython -Label "Summarize $name" -Arguments @(
    "train/summarize_experiment.py", "--experiment-dir", $resultDir
)
