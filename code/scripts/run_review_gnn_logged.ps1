param(
    [ValidateSet("All", "Deployment", "FullMatrix")]
    [string]$GnnSet = "All",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot "results/review/_runner_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$transcriptPath = Join-Path $logDir "strict_gnn_${GnnSet}_${runId}.transcript.log"
$nativeLogPath = Join-Path $logDir "strict_gnn_${GnnSet}_${runId}.native.log"
$statusPath = Join-Path $logDir "latest_status.json"
$startedAt = (Get-Date).ToUniversalTime().ToString("o")

function Write-RunnerStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [string]$Message = "",
        [Nullable[int]]$ExitCode = $null
    )
    $payload = [ordered]@{
        state = $State
        gnn_set = $GnnSet
        process_id = $PID
        started_at_utc = $startedAt
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        transcript = $transcriptPath
        native_log = $nativeLogPath
        exit_code = $ExitCode
        message = $Message
    }
    $payload | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-RunnerStatus -State "running" -Message "Strict review GNN queue started."
Start-Transcript -LiteralPath $transcriptPath -Force | Out-Null

try {
    $runnerPath = Join-Path $PSScriptRoot "run_review_baselines.ps1"
    # Do not merge stderr into the PowerShell success stream. common.ps1 sets
    # ErrorActionPreference=Stop, and Windows PowerShell 5.1 would otherwise
    # promote harmless native-library warnings to terminating RemoteException.
    if ($Force) {
        & $runnerPath -SkipTabular -GnnSet $GnnSet -Force |
            Tee-Object -FilePath $nativeLogPath -Append
    }
    else {
        & $runnerPath -SkipTabular -GnnSet $GnnSet |
            Tee-Object -FilePath $nativeLogPath -Append
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Review GNN runner exited with code $LASTEXITCODE."
    }
    Write-RunnerStatus -State "completed" -Message "Strict review GNN queue completed." -ExitCode 0
}
catch {
    Write-RunnerStatus -State "failed" -Message $_.Exception.Message -ExitCode 1
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
