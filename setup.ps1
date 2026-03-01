# Quick Setup Script for Windows PowerShell
# Script thiết lập nhanh cho Windows

Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host "="
Write-Host "PhoBERT Contract Processing - Quick Setup"
Write-Host "=" -NoNewline; for($i=0; $i -lt 59; $i++) { Write-Host "=" -NoNewline }; Write-Host "="

# Check Python
Write-Host "`n[1/5] Checking Python installation..."
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $pythonVersion = python --version
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n[2/5] Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[3/5] Activating virtual environment..."
.\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Install requirements
Write-Host "`n[4/5] Installing dependencies..."
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Create necessary directories
Write-Host "`n[5/5] Creating directories..."
$dirs = @("outputs", "logs", "data/raw", "data/processed")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green

# Summary
Write-Host "`n" + ("=" * 60)
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ("=" * 60)
Write-Host "`nNext steps:"
Write-Host "1. Prepare your data: python main.py prepare-data --input-file data/sample_data.json"
Write-Host "2. Train model: python main.py train"
Write-Host "3. Make predictions: python main.py predict --checkpoint outputs/best_model"
Write-Host "`nFor more info: python main.py --help"
