param(
    [switch]$SkipTabular,
    [switch]$SkipGnn,
    [ValidateSet("All", "Deployment", "FullMatrix")]
    [string]$GnnSet = "All",
    [int]$BootstrapReplicates = 2000,
    [int]$BootstrapSeed = 20260809,
    [switch]$Force
)

. "$PSScriptRoot\common.ps1"

$reviewDir = "train/experiments/review"
Invoke-TwoPython -Label "Generate and audit review split manifests" -Arguments @(
    "data/make_review_splits.py"
)
$tabularConfigs = @(
    "R11_dynamic_only_xgb_random.yaml",
    "R12_dynamic_ehull_xgb_random.yaml",
    "R13_dynamic_only_xgb_composition.yaml",
    "R14_dynamic_ehull_xgb_composition.yaml",
    "R15_dynamic_only_xgb_chemical.yaml",
    "R16_dynamic_ehull_xgb_chemical.yaml",
    "R17_dynamic_only_xgb_structure.yaml",
    "R18_dynamic_ehull_xgb_structure.yaml"
)
$deploymentGnnConfigs = @(
    "R01_dynamic_only_alignn_random.yaml",
    "R02_dynamic_ehull_alignn_random.yaml",
    "R03_dynamic_only_alignn_composition.yaml",
    "R04_dynamic_ehull_alignn_composition.yaml",
    "R05_dynamic_only_alignn_chemical.yaml",
    "R06_dynamic_ehull_alignn_chemical.yaml",
    "R07_dynamic_only_alignn_structure.yaml",
    "R08_dynamic_ehull_alignn_structure.yaml"
)
$fullMatrixGnnConfigs = @(
    "R21_single_joint_alignn_random.yaml",
    "R22_equal_mtl_alignn_random.yaml",
    "R23_full_mtl_alignn_random.yaml",
    "R24_full_no_logic_alignn_random.yaml",
    "R25_single_joint_alignn_composition.yaml",
    "R26_equal_mtl_alignn_composition.yaml",
    "R27_full_mtl_alignn_composition.yaml",
    "R28_full_no_logic_alignn_composition.yaml",
    "R29_single_joint_alignn_chemical.yaml",
    "R30_equal_mtl_alignn_chemical.yaml",
    "R31_full_mtl_alignn_chemical.yaml",
    "R32_full_no_logic_alignn_chemical.yaml",
    "R33_single_joint_alignn_structure.yaml",
    "R34_equal_mtl_alignn_structure.yaml",
    "R35_full_mtl_alignn_structure.yaml",
    "R36_full_no_logic_alignn_structure.yaml"
)

function Invoke-ReviewExperiment {
    param(
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][ValidateSet("gnn", "tabular")][string]$Kind
    )
    $entry = if ($Kind -eq "gnn") { "train/train_gnn.py" } else { "train/train_tabular.py" }
    $configPath = "$reviewDir/$Config"
    $arguments = @($entry, "--experiment", $configPath)
    if ($Force) { $arguments += "--force" }
    Invoke-TwoPython -Label "Review baseline $Config" -Arguments $arguments

    $name = [System.IO.Path]::GetFileNameWithoutExtension($Config)
    Invoke-TwoPython -Label "Summarize $name" -Arguments @(
        "train/summarize_experiment.py",
        "--experiment-dir",
        "D:/ck/two/results/review/$name"
    )
}

if (-not $SkipTabular) {
    foreach ($config in $tabularConfigs) {
        Invoke-ReviewExperiment -Config $config -Kind tabular
    }
    Invoke-TwoPython -Label "Summarize paired review deployment baselines" -Arguments @(
        "train/summarize_review_baselines.py",
        "--project-root", "D:/ck/two",
        "--bootstrap-replicates", "$BootstrapReplicates",
        "--bootstrap-seed", "$BootstrapSeed"
    )
}

if (-not $SkipGnn) {
    $gnnConfigs = switch ($GnnSet) {
        "Deployment" { $deploymentGnnConfigs }
        "FullMatrix" { $fullMatrixGnnConfigs }
        default { $deploymentGnnConfigs + $fullMatrixGnnConfigs }
    }
    foreach ($config in $gnnConfigs) {
        Invoke-ReviewExperiment -Config $config -Kind gnn
    }
    $summaryScope = switch ($GnnSet) {
        "Deployment" { "deployment" }
        "FullMatrix" { "architecture" }
        default { "all" }
    }
    Invoke-TwoPython -Label "Fail-closed five-seed strict GNN summary" -Arguments @(
        "train/summarize_review_gnn.py",
        "--project-root", "D:/ck/two",
        "--scope", $summaryScope,
        "--bootstrap-replicates", "$BootstrapReplicates",
        "--bootstrap-seed", "$BootstrapSeed"
    )
}
