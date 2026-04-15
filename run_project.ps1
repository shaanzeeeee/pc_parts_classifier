param (
    [switch]$SkipDownload
)

Write-Host "========================================="
Write-Host "  Hardware-Vision Execution Pipeline"
Write-Host "========================================="

# Ensure we are executing within the virtual environment environment context.
$env:PYTHONUNBUFFERED = "1"

if (-Not $SkipDownload) {
    Write-Host "`n[1/3] Starting Data Acquisition Module..."
    .\venv\Scripts\python.exe dataset_creation\data_acquisition.py
    if ($LASTEXITCODE -ne 0) { Write-Error "Data Acquisition Failed"; exit $LASTEXITCODE }
} else {
    Write-Host "`n[1/3] Skipping Data Acquisition Module..."
}

Write-Host "`n[2/3] Starting Data Preprocessing Module..."
.\venv\Scripts\python.exe dataset_creation\data_preprocessing.py
if ($LASTEXITCODE -ne 0) { Write-Error "Data Preprocessing Failed"; exit $LASTEXITCODE }

Write-Host "`n[3/3] Starting Model Training & Evaluation Module..."
.\venv\Scripts\python.exe src\main.py
if ($LASTEXITCODE -ne 0) { Write-Error "Training Pipeline Failed"; exit $LASTEXITCODE }

Write-Host "`n========================================="
Write-Host "  Pipeline Execution Complete."
Write-Host "========================================="
