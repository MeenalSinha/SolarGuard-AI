#!/bin/bash

# Solar Panel Damage Detection System - Setup Script
# This script automates the installation and setup process

echo "☀️ Solar Panel Damage Detection System - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "❌ Error: Python 3.8+ required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"
echo ""

# Create virtual environment
echo "🔨 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ Pip upgraded"
echo ""

# Install requirements
echo "📥 Installing dependencies..."
echo "   This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p models
mkdir -p datasets
mkdir -p data/train
mkdir -p data/val
echo "✅ Directories created"
echo ""

# Check for Kaggle credentials
echo "🔑 Checking Kaggle API credentials..."
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "✅ Kaggle credentials found"
    chmod 600 "$HOME/.kaggle/kaggle.json"
else
    echo "⚠️  Kaggle credentials not found"
    echo ""
    echo "To download datasets, you need Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Save kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "You can skip this and train with your own data."
fi
echo ""

echo "=============================================="
echo "✅ Setup Complete!"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "Option 1: Launch Web Interface"
echo "  $ source venv/bin/activate"
echo "  $ streamlit run solar_panel_detection.py"
echo ""
echo "Option 2: Download Data & Train Model"
echo "  $ source venv/bin/activate"
echo "  $ python solar_panel_detection.py --download-data"
echo "  $ python solar_panel_detection.py --train --epochs 50"
echo ""
echo "Option 3: Analyze Single Image"
echo "  $ source venv/bin/activate"
echo "  $ python solar_panel_detection.py --predict image.jpg"
echo ""
echo "📖 For more information, see README.md"
echo "=============================================="
