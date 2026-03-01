#!/bin/bash
# Quick Setup Script for Linux/Mac
# Script thiết lập nhanh cho Linux/Mac

echo "============================================================"
echo "PhoBERT Contract Processing - Quick Setup"
echo "============================================================"

# Check Python
echo -e "\n[1/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "✓ Found: $python_version"
else
    echo "✗ Python not found! Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install requirements
echo -e "\n[4/5] Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo -e "\n[5/5] Creating directories..."
mkdir -p outputs logs data/raw data/processed
echo "✓ Directories created"

# Summary
echo -e "\n============================================================"
echo "Setup completed successfully!"
echo "============================================================"
echo -e "\nNext steps:"
echo "1. Prepare your data: python main.py prepare-data --input-file data/sample_data.json"
echo "2. Train model: python main.py train"
echo "3. Make predictions: python main.py predict --checkpoint outputs/best_model"
echo -e "\nFor more info: python main.py --help"
