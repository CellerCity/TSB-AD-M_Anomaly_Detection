# Ablation sweep over (context_length, horizon) for TimesFM zero-shot anomaly detection.
#
# Each cell runs the forecast once and evaluates all 16 (score x agg) combinations.
# Output: one CSV per cell + one final summary CSV across the whole sweep.
#
# Native Windows PowerShell. Run from a PowerShell session with your conda env active.
#
# Usage:
#   .\run_ablation.ps1 -Dataset SMD
#   .\run_ablation.ps1 -Dataset Exathlon -Force
#   .\run_ablation.ps1 -Dataset SMD -DataPattern ".\custom\path\*.csv"

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Dataset,

    [string]$DataPattern = "",
    [string]$DataRoot    = ".\mTSBench",
    [switch]$Force
)

# -------------------- DEFAULT CONFIG --------------------
$Contexts    = @(256, 512, 1024)
$Horizons    = @(5, 10, 30, 50, 100)
$ResultsRoot = "ablation_results"
$LogsRoot    = "ablation_logs"
$SummaryDir  = "ablation_summaries"

# Derive data pattern from dataset name if not overridden.
if ($DataPattern -eq "") {
    $DataPattern = Join-Path (Join-Path $DataRoot $Dataset) "*test.csv"
}

# Per-dataset output dirs so different datasets don't collide.
$ResultsDir = Join-Path $ResultsRoot $Dataset
$LogsDir    = Join-Path $LogsRoot    $Dataset
$SummaryCsv = Join-Path $SummaryDir  "$($Dataset)_combos_summary.csv"

New-Item -ItemType Directory -Force -Path $ResultsDir, $LogsDir, $SummaryDir | Out-Null

# -------------------- PRE-FLIGHT CHECKS --------------------
# Confirm Python sees its required packages and the GPU.
& python -c "import numpy, torch, timesfm; print('python OK; CUDA available:', torch.cuda.is_available())" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python environment missing required packages." -ForegroundColor Red
    Write-Host "  Make sure your conda env is activated before running this script." -ForegroundColor Red
    exit 1
}

# Confirm the glob matches something.
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
Write-Host "  results dir:    $ResultsDir\"
Write-Host "  logs dir:       $LogsDir\"
Write-Host "  summary csv:    $SummaryCsv"
Write-Host "  force re-run:   $Force"
Write-Host "  (each cell evaluates all 16 score x agg combos in one forecast)"
Write-Host "================================================================"

# -------------------- RUN GRID --------------------
$Total     = $Contexts.Count * $Horizons.Count
$Counter   = 0
$Succeeded = @()
$Skipped   = @()
$Failed    = @()
$StartTime = Get-Date

foreach ($ctx in $Contexts) {
  foreach ($hor in $Horizons) {
        $Counter++
        $Tag     = "c${ctx}_h${hor}_all"
        $OutCsv  = Join-Path $ResultsDir "$Tag.csv"
        $LogFile = Join-Path $LogsDir    "$Tag.log"

        Write-Host ""
        Write-Host "[$Counter/$Total] $Dataset context=$ctx horizon=$hor"

        if ((Test-Path $OutCsv) -and (-not $Force)) {
            Write-Host "  -> Output exists, skipping (use -Force to re-run): $OutCsv"
            $Skipped += $Tag
            continue
        }

        $CellStart = Get-Date
        & python timesFM_all_combos.py `
            --data_pattern   $DataPattern `
            --context_length $ctx `
            --horizon        $hor `
            --save_path      $OutCsv `
            *>&1 | Out-File -FilePath $LogFile -Encoding UTF8
        $Status  = $LASTEXITCODE
        $Elapsed = [int]((Get-Date) - $CellStart).TotalSeconds

        if (($Status -eq 0) -and (Test-Path $OutCsv)) {
            Write-Host "  -> OK in ${Elapsed}s   ($OutCsv)"
            $Succeeded += $Tag
        }
        else {
            Write-Host "  -> FAILED (exit=$Status, see $LogFile)" -ForegroundColor Red
            $Failed += $Tag
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

# Aggregate every cell output we have on disk (succeeded + skipped-but-present).
$AggInputs = @(Get-ChildItem -Path (Join-Path $ResultsDir "*_all.csv"))
if ($AggInputs.Count -eq 0) {
    Write-Host "No output files to aggregate. Exiting."
    exit 1
}

Write-Host ""
Write-Host "Aggregating $($AggInputs.Count) cell files into $SummaryCsv ..."
$InputPaths = $AggInputs | ForEach-Object { $_.FullName }
& python average_combos.py `
    --inputs @InputPaths `
    --output $SummaryCsv

Write-Host ""
Write-Host "Done. Final summary: $SummaryCsv"