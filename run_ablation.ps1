# Ablation sweep over (context_length, horizon) for TimesFM zero-shot anomaly detection.
#
# - Configurable per dataset (SMD, Exathlon, etc).
# - Skips cells whose output CSV already exists (resume-friendly after a crash).
# - Logs each cell's stdout/stderr to logs/.
# - Continues past individual cell failures.
# - Calls average_methods.py at the end to produce one summary CSV per dataset.
#
# Usage:
#   .\run_ablation.ps1 -Dataset SMD
#   .\run_ablation.ps1 -Dataset SMD -ScoreMethod mse
#   .\run_ablation.ps1 -Dataset SMD -ScoreMethods "interval","mse"
#   .\run_ablation.ps1 -Dataset Exathlon -Force
#   .\run_ablation.ps1 -Dataset SMD -DataPattern ".\custom\path\*.csv"

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,

    [string]$DataPattern = "",
    [string]$DataRoot = ".\mTSBench",
    [string]$ScoreMethod = "",
    [string[]]$ScoreMethods = @("interval"),
    [string]$AggMethod = "l2",
    [switch]$Force
)

# Allow -ScoreMethod (singular) as a shortcut for one method.
if ($ScoreMethod -ne "") {
    $ScoreMethods = @($ScoreMethod)
}

# -------------------- DEFAULT CONFIG --------------------
$Contexts = @(256, 512, 1024)
$Horizons = @(5, 10, 30, 50, 100)
$ResultsRoot = "ablation_results"
$LogsRoot    = "ablation_logs"
$SummaryDir  = "ablation_summaries"

# Derive data pattern from dataset name if not overridden.
if ($DataPattern -eq "") {
    $DataPattern = Join-Path (Join-Path $DataRoot $Dataset) "*test.csv"
}

# Per-dataset output dirs so sweeps don't collide.
$ResultsDir = Join-Path $ResultsRoot $Dataset
$LogsDir    = Join-Path $LogsRoot $Dataset
$SummaryCsv = Join-Path $SummaryDir "$($Dataset)_summary.csv"

New-Item -ItemType Directory -Force -Path $ResultsDir, $LogsDir, $SummaryDir | Out-Null

# Confirm the glob actually matches something before we start.
$Matched = @(Get-ChildItem -Path $DataPattern -ErrorAction SilentlyContinue)
if ($Matched.Count -eq 0) {
    Write-Host "ERROR: data_pattern '$DataPattern' matched 0 files." -ForegroundColor Red
    Write-Host "  Check that the dataset exists under $DataRoot\$Dataset\" -ForegroundColor Red
    Write-Host "  Or pass -DataPattern explicitly." -ForegroundColor Red
    exit 1
}

Write-Host "================================================================"
Write-Host "Ablation sweep: $Dataset"
Write-Host "  data pattern:   $DataPattern"
Write-Host "  files matched:  $($Matched.Count)"
Write-Host "  contexts:       $($Contexts -join ' ')"
Write-Host "  horizons:       $($Horizons -join ' ')"
Write-Host "  score_methods:  $($ScoreMethods -join ' ')"
Write-Host "  agg_method:     $AggMethod"
Write-Host "  results dir:    $ResultsDir\"
Write-Host "  logs dir:       $LogsDir\"
Write-Host "  summary csv:    $SummaryCsv"
Write-Host "  force re-run:   $Force"
Write-Host "================================================================"

# Quick sanity check that Python sees its packages.
& python -c "import numpy, torch, timesfm; print('python OK, torch CUDA available:', torch.cuda.is_available())" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python environment is missing required packages. Activate the conda env first." -ForegroundColor Red
    exit 1
}

# -------------------- RUN GRID --------------------
$Total = $Contexts.Count * $Horizons.Count * $ScoreMethods.Count
$Counter = 0
$Succeeded = @()
$Skipped   = @()
$Failed    = @()

# Parallel arrays for the summary call.
$SummaryFiles = @()
$SummaryCtx   = @()
$SummaryHor   = @()
$SummaryScore = @()

$StartTime = Get-Date

foreach ($sm in $ScoreMethods) {
  foreach ($ctx in $Contexts) {
    foreach ($hor in $Horizons) {
        $Counter++
        $Tag     = "c${ctx}_h${hor}_${sm}"
        $OutCsv  = Join-Path $ResultsDir "$Tag.csv"
        $LogFile = Join-Path $LogsDir    "$Tag.log"

        Write-Host ""
        Write-Host "[$Counter/$Total] $Dataset context=$ctx horizon=$hor score=$sm"

        if ((Test-Path $OutCsv) -and (-not $Force)) {
            Write-Host "  -> Output exists, skipping (use -Force to re-run): $OutCsv"
            $Skipped      += $Tag
            $SummaryFiles += $OutCsv
            $SummaryCtx   += $ctx
            $SummaryHor   += $hor
            $SummaryScore += $sm
            continue
        }

        $CellStart = Get-Date
        & python timesFM_modularised.py `
            --data_pattern   $DataPattern `
            --context_length $ctx `
            --horizon        $hor `
            --score_method   $sm `
            --agg_method     $AggMethod `
            --save_path      $OutCsv `
            *>&1 | Out-File -FilePath $LogFile -Encoding UTF8
        $Status  = $LASTEXITCODE
        $Elapsed = [int]((Get-Date) - $CellStart).TotalSeconds

        if (($Status -eq 0) -and (Test-Path $OutCsv)) {
            Write-Host "  -> OK in ${Elapsed}s   ($OutCsv)"
            $Succeeded    += $Tag
            $SummaryFiles += $OutCsv
            $SummaryCtx   += $ctx
            $SummaryHor   += $hor
            $SummaryScore += $sm
        }
        else {
            Write-Host "  -> FAILED (exit=$Status, see $LogFile)" -ForegroundColor Red
            $Failed += $Tag
        }
    }
  }
}

$TotalTime = [int]((Get-Date) - $StartTime).TotalSeconds

# -------------------- SUMMARY --------------------
Write-Host ""
Write-Host "================================================================"
Write-Host "Sweep complete for $Dataset in $([int]($TotalTime / 60))m $($TotalTime % 60)s"
Write-Host "  succeeded: $($Succeeded.Count) / $Total"
Write-Host "  skipped:   $($Skipped.Count) (output already existed)"
Write-Host "  failed:    $($Failed.Count)"
if ($Failed.Count -gt 0) {
    Write-Host "  failed cells: $($Failed -join ' ')" -ForegroundColor Yellow
}
Write-Host "================================================================"

if ($SummaryFiles.Count -eq 0) {
    Write-Host "No output files to summarize. Exiting."
    exit 1
}

Write-Host ""
Write-Host "Aggregating $($SummaryFiles.Count) cells into $SummaryCsv ..."
& python average_methods.py `
    --inputs        @SummaryFiles `
    --contexts      @SummaryCtx `
    --horizons      @SummaryHor `
    --score_methods @SummaryScore `
    --output        $SummaryCsv

Write-Host ""
Write-Host "Done. Final summary: $SummaryCsv"
