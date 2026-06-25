param(
    [string]$Mode = "dual2",
    [string]$ContextMode = "causal",
    [string]$MaxEpochs = "120",
    [string[]]$Dual2Channels = @("Fp1", "Fp2")
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OfflineDir = Join-Path $ProjectRoot "offline_training"
$SubjectsFile = Join-Path $OfflineDir "subjects_template.json"

Write-Host "Project root: $ProjectRoot"
Write-Host "Offline dir:  $OfflineDir"
Write-Host "Mode:         $Mode"
Write-Host "Context:      $ContextMode"
Write-Host "Max epochs:   $MaxEpochs"
if ($Mode -eq "dual2") {
    Write-Host "Dual2 chans:  $($Dual2Channels -join ', ')"
}

Push-Location $OfflineDir
try {
    Write-Host ""
    Write-Host "Step 1/2: extracting EEG features..."
    if ($Mode -eq "dual2") {
        if ($Dual2Channels.Count -ne 2) {
            throw "Dual2Channels must contain exactly two channel names, for example: -Dual2Channels Fp1,Fp2"
        }
        python 01_extract_features_LIGHT_FIR_vscode.py --subjects-file $SubjectsFile --mode $Mode --max-epochs $MaxEpochs --dual2-channels $Dual2Channels[0] $Dual2Channels[1]
    }
    else {
        python 01_extract_features_LIGHT_FIR_vscode.py --subjects-file $SubjectsFile --mode $Mode --max-epochs $MaxEpochs
    }

    Write-Host ""
    Write-Host "Step 2/2: training sleep stage model..."
    python 02_train_compare_models_vscode.py --mode $Mode --context-mode $ContextMode

    Write-Host ""
    Write-Host "Done."
    Write-Host "Model: offline_training/saved_models/sleep_stage_${Mode}_${ContextMode}_global.joblib"
}
finally {
    Pop-Location
}
