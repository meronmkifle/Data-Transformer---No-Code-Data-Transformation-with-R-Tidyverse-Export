@echo off
REM Data Transformer Setup Script for Windows
REM This script will set up your development environment

echo.
echo ============================================
echo   Data Transformer - Windows Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3 not found. Please install Python 3.9 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Git not found. Installing anyway, but you may need Git from https://git-scm.com/
    echo.
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo ✓ Pip upgraded
echo.

REM Install requirements
echo Installing dependencies...
echo   - streamlit
echo   - pandas
echo   - openpyxl
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Success message
echo ============================================
echo ✓ Setup Complete!
echo ============================================
echo.
echo To start the app, run:
echo   venv\Scripts\activate.bat
echo   streamlit run data_transformer_app.py
echo.
echo The app will open at http://localhost:8501
echo.
echo To deactivate the virtual environment later:
echo   deactivate
echo.
pause
