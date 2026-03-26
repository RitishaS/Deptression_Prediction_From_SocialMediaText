@echo off
REM Quick Start Script for Windows - GloVe Depression Classifier

echo.
echo ============================================================
echo 🚀 QUICK START: GloVe Depression Classifier (Windows)
echo ============================================================
echo.

REM Check Python
echo ⏳ Checking Python installation...
python --version

if errorlevel 1 (
    echo ❌ Python not found! Install Python first.
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo 📦 Installing required packages...
pip install -r requirements_glove.txt

if errorlevel 1 (
    echo ❌ Failed to install requirements!
    pause
    exit /b 1
)

REM Run main pipeline
echo.
echo ✨ Starting the pipeline...
echo ============================================================
python src/GloVe_Depression_Classifier.py

echo.
echo ✅ Complete! Check output above for results.
pause
