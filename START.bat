@echo off
REM Depression Level Analyzer - Startup Script for Windows

echo.
echo ================================================================
echo   Depression Level Analyzer - Startup Script
echo ================================================================
echo.

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)"
if errorlevel 1 (
    echo [!] Virtual environment not activated. Activating...
    call .venv\Scripts\activate.bat
)

echo [+] Virtual environment is active
echo.

REM Check if required packages are installed
echo [*] Checking required packages...
python -c "import flask; import flask_cors; import nltk; import sklearn" 2>nul
if errorlevel 1 (
    echo [!] Installing required packages...
    pip install flask flask-cors nltk scikit-learn pandas numpy
)

echo.
echo [+] All dependencies are installed
echo.

REM Start the backend server
echo ================================================================
echo [*] Starting Depression Level Analyzer Backend...
echo ================================================================
echo.
echo URL: http://localhost:5000
echo.
echo Press CTRL+C to stop the server
echo.

cd backend
python app.py

pause
