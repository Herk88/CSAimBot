@echo off
REM ================================================================
REM  CSAimbot launcher
REM  - Activates a local virtual environment if present
REM  - Falls back to system Python otherwise
REM  - Starts the GUI controller so you can tweak settings
REM ================================================================

setlocal ENABLEDELAYEDEXPANSION

REM Change into the repository directory (the folder that holds this BAT).
cd /d "%~dp0"

REM Ensure we can find python after activation.
set "PYTHON_CMD=python"

REM Prefer a virtual environment named .venv (common in this project).
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
) else (
    echo [INFO] No local virtual environment found. Using system Python.
)

REM If python.exe is bundled inside the venv, prefer it explicitly.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=%CD%\venv\Scripts\python.exe"
)

REM Validate that a Python interpreter is reachable.
set "PYTHON_EXEC=%PYTHON_CMD%"
if not exist "%PYTHON_EXEC%" (
    where %PYTHON_CMD% >nul 2>&1
    if errorlevel 1 (
        where py >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Could not locate a Python interpreter. Please install Python or create a virtual environment.
            goto :EOF
        ) else (
            echo [INFO] Falling back to Python launcher.
            set "PYTHON_EXEC=py"
        )
    ) else (
        set "PYTHON_EXEC=%PYTHON_CMD%"
    )
) else (
    set "PYTHON_EXEC=%PYTHON_CMD%"
)

if /I "%PYTHON_EXEC%"=="py" (
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Could not locate a Python interpreter. Please install Python or create a virtual environment.
        goto :EOF
    )
)

REM Launch the CSAimbot GUI.
echo [INFO] Starting CSAimbot GUI...
"%PYTHON_EXEC%" "%~dp0Run_Me.py" --gui

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    echo [INFO] CSAimbot stopped cleanly.
) else (
    echo [ERROR] CSAimbot exited with code %EXIT_CODE%.
)

REM Keep the window open long enough to read messages.
echo.
pause
