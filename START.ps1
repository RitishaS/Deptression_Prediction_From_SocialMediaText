# Depression Level Analyzer - Startup Script for PowerShell

Write-Host ""
Write-Host "================================================================"
Write-Host "   Depression Level Analyzer - Startup Script (PowerShell)"
Write-Host "================================================================"
Write-Host ""

# Check if virtual environment is activated
$venvActive = $env:VIRTUAL_ENV -ne $null

if (-not $venvActive) {
    Write-Host "[!] Virtual environment not activated. Activating..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    Write-Host "[+] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[+] Virtual environment is already active" -ForegroundColor Green
}

Write-Host ""

# Check if required packages are installed
Write-Host "[*] Checking required packages..." -ForegroundColor Cyan

$packagesInstalled = $true
try {
    python -c "import flask; import flask_cors; import nltk; import sklearn" 2>$null
} catch {
    $packagesInstalled = $false
}

if (-not $packagesInstalled) {
    Write-Host "[!] Installing required packages..." -ForegroundColor Yellow
    pip install flask flask-cors nltk scikit-learn pandas numpy
}

Write-Host ""
Write-Host "[+] All dependencies are installed" -ForegroundColor Green
Write-Host ""

# Start the backend server
Write-Host "================================================================"
Write-Host "[*] Starting Depression Level Analyzer Backend..." -ForegroundColor Cyan
Write-Host "================================================================"
Write-Host ""
Write-Host "URL: http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Set-Location backend
python app.py

Read-Host "Press Enter to exit"
