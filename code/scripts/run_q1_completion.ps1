param(
    [ValidateSet("Minimum3", "Full5")]
    [string]$Plan = "Full5",
    [switch]$Force,
    [switch]$DryRun
)

# Completes only Q1's missing fair fixed-IID single-task baselines.
# Existing M04 seeds 2025-2027 and all M06 Full-MTL seeds are deliberately not run.
. "$PSScriptRoot\common.ps1"

$baseConfigs = @(
    "train/experiments/q1/Q1_thermo_only_alignn_iid.yaml",
    "train/experiments/q1/Q1_dynamic_only_alignn_iid.yaml"
)
$seeds = if ($Plan -eq "Minimum3") { @(2025, 2026, 2027) } else { @(2025, 2026, 2027, 2028, 2029) }
$jobs = @()
foreach ($experiment in $baseConfigs) {
    foreach ($seed in $seeds) {
        $jobs += [PSCustomObject]@{ Experiment = $experiment; Seed = $seed }
    }
}
if ($Plan -eq "Full5") {
    foreach ($seed in @(2028, 2029)) {
        $jobs += [PSCustomObject]@{
            Experiment = "train/experiments/q1/Q1_joint_only_alignn_iid_2028_2029.yaml"
            Seed = $seed
        }
    }
}

if ($DryRun) {
    foreach ($job in $jobs) {
        Write-Host "DRY RUN: $($job.Experiment), seed $($job.Seed)"
    }
    Write-Host "Q1 completion plan $Plan dry run: $($jobs.Count) training jobs."
    exit 0
}

$completedConfigs = @{}
foreach ($job in $jobs) {
    $arguments = @("train/train_gnn.py", "--experiment", $job.Experiment, "--seed", "$($job.Seed)")
    if ($Force) { $arguments += "--force" }
    Invoke-TwoPython -Label "Q1 completion: $($job.Experiment), seed $($job.Seed)" -Arguments $arguments
    $completedConfigs[$job.Experiment] = $true
}

foreach ($experiment in $completedConfigs.Keys) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($experiment)
    Invoke-TwoPython -Label "Summarize Q1 run $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir", "D:/ck/two/results/q1/$name"
    )
}

Write-Host "Q1 completion plan $Plan finished. Existing completed seeds were skipped unless -Force was supplied."
