@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo VOICE GUARDIAN - SETUP ^& INTEGRATION SCRIPT (Windows)
echo ========================================================================
echo.

REM Check Python version
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.9+
    exit /b 1
)
python --version

REM Install dependencies
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo Dependencies installed successfully!

REM Create training_data directory
echo.
echo Creating training_data directory...
if not exist "training_data" mkdir training_data
if exist "training_data" (
    echo training_data directory created/verified
) else (
    echo ERROR: Failed to create training_data directory
    exit /b 1
)

REM Create pretrained_models directory
echo.
echo Ensuring pretrained_models directory exists...
if not exist "pretrained_models" mkdir pretrained_models

REM Verify key files
echo.
echo Verifying key files...
set "missing=0"
for %%F in (main.py index.html voice_guardian.py voice_guardian_enhanced.py frontend_server.py requirements.txt) do (
    if exist "%%F" (
        echo   [OK] %%F
    ) else (
        echo   [MISSING] %%F
        set /a "missing+=1"
    )
)

if !missing! gtr 0 (
    echo.
    echo ERROR: Some files are missing!
    exit /b 1
)

echo.
echo ========================================================================
echo SETUP COMPLETE!
echo ========================================================================
echo.
echo Next steps:
echo.
echo 1. Add training data:
echo    copy your_voice_samples.wav training_data\
echo.
echo 2. Start the backend (Command Prompt 1):
echo    python main.py
echo.
echo 3. Start the frontend (Command Prompt 2):
echo    python frontend_server.py
echo.
echo 4. Open in browser:
echo    http://localhost:3000
echo.
echo ========================================================================
echo.
pause
