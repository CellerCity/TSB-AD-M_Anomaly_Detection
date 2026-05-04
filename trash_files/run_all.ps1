# run_all.ps1
# PowerShell wrapper to launch the parallel TSB-AD-M anomaly detection benchmark.
# Runs CPU pool (5 unsupervised models) and GPU pool (OmniAnomaly) simultaneously.
#
# Usage from PowerShell (in this directory):
#     .\run_all.ps1                     # full Eval-set run, all 6 models
#     .\run_all.ps1 -SmokeTest          # 1 file per dataset, all 6 models
#     .\run_all.ps1 -Models PCA,IForest # only run specific models
#     .\run_all.ps1 -Fresh              # delete prior results before starting
#
# If you get "running scripts is disabled on this system", run this once
# in an admin PowerShell:
#     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

[CmdletBinding()]
param(
    # Models to run. Default = all six.
    [string[]]$Models = @("PCA", "IForest", "CBLOF", "RobustPCA", "KMeansAD", "OmniAnomaly"),

    # Path to TSB-AD-M data directory.
    [string]$DataDir = ".\TSB-AD-M",

    # Path to TSB-AD-M-Eva.csv (the Eval-set file list).
    [string]$EvalCsv = ".\File_List\TSB-AD-M-Eva.csv",

    # Where to write outputs.
    [string]$OutDir = ".\results",

    # CPU pool size. Default lets launcher.py auto-detect.
    [int]$Workers = 0,

    # GPU pool size. Default 1 (safest, prevents GPU OOM).
    [int]$GpuWorkers = 1,

    # Run only the first N files per dataset (smoke testing).
    [int]$LimitPerDataset = 0,

    # Skip the Eval-set filter; run on all files in DataDir.
    [switch]$NoEvalFilter,

    # Delete prior per-file CSVs before starting.
    [switch]$Fresh,

    # Smoke test shortcut: 1 file per dataset.
    [switch]$SmokeTest,

    # Path to python executable. Defaults to whatever's on PATH.
    [string]$Python = "python"
)

# Resolve to absolute paths so the script works from any directory
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LauncherPy = Join-Path $ScriptRoot "launcher.py"

if (-not (Test-Path $LauncherPy)) {
    Write-Error "launcher.py not found at $LauncherPy"
    exit 1
}

# Build the python argument list
$pyArgs = @(
    $LauncherPy,
    "--models"
) + $Models + @(
    "--data-dir", $DataDir,
    "--eval-csv", $EvalCsv,
    "--out-dir", $OutDir,
    "--gpu-workers", $GpuWorkers
)

if ($Workers -gt 0) {
    $pyArgs += @("--workers", $Workers)
}

if ($SmokeTest) {
    $pyArgs += @("--limit-per-dataset", "1")
} elseif ($LimitPerDataset -gt 0) {
    $pyArgs += @("--limit-per-dataset", $LimitPerDataset)
}

if ($NoEvalFilter) {
    $pyArgs += "--no-eval-filter"
}

if ($Fresh) {
    $pyArgs += "--fresh"
}

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "TSB-AD-M Parallel Launcher" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Python:       $Python"
Write-Host "Launcher:     $LauncherPy"
Write-Host "Models:       $($Models -join ', ')"
Write-Host "Data dir:     $DataDir"
Write-Host "Out dir:      $OutDir"
if ($SmokeTest)       { Write-Host "Mode:         SMOKE TEST (1 file/dataset)" -ForegroundColor Yellow }
elseif ($LimitPerDataset -gt 0) { Write-Host "Mode:         Limited ($LimitPerDataset files/dataset)" -ForegroundColor Yellow }
else                  { Write-Host "Mode:         Full Eval-set run" }
if ($Fresh)           { Write-Host "Cleanup:      DELETING prior results" -ForegroundColor Yellow }
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Run it
& $Python @pyArgs
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "Done. Results in $OutDir" -ForegroundColor Green
} else {
    Write-Host "Launcher exited with code $exitCode" -ForegroundColor Red
}
Write-Host "===========================================" -ForegroundColor Cyan

exit $exitCode
