@echo off
rem ============================================================
rem  Whisper + OpenVINO one-click setup (Windows / Intel Arc)
rem
rem  First run: right-click this file, "Run as administrator".
rem
rem  NOTE: this file is intentionally ASCII-only with CRLF endings.
rem        All Chinese messages are printed by the Python program.
rem ============================================================

setlocal enabledelayedexpansion
chcp 65001 >nul
title Whisper setup - OpenVINO on Intel Arc

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo  ============================================================
echo    Whisper high-performance transcription - setup
echo    Target directory: %ROOT%
echo  ============================================================
echo.

rem ---------- 1. locate Python ----------
echo [1/5] Looking for Python...
set "PYEXE="
for %%P in (
    "%ROOT%venv\Scripts\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Program Files\Python313\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
) do (
    if not defined PYEXE if exist %%P set "PYEXE=%%~P"
)
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE set "PYEXE=%%i"
    )
)
if not defined PYEXE (
    echo   [ERROR] Python 3.10+ not found.
    echo           Install it from https://www.python.org/downloads/
    echo           and be sure to tick "Add python.exe to PATH".
    pause
    exit /b 1
)
echo   Using: %PYEXE%
"%PYEXE%" -c "import sys; assert sys.version_info>=(3,10), sys.version"
if errorlevel 1 (
    echo   [ERROR] Python is too old, 3.10 or newer is required.
    pause
    exit /b 1
)

rem ---------- 2. virtual environment ----------
echo.
echo [2/5] Preparing virtual environment...
if not exist "%ROOT%venv\Scripts\python.exe" (
    "%PYEXE%" -m venv "%ROOT%venv"
    if errorlevel 1 (
        echo   [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
    echo   Created.
) else (
    echo   Already exists, skipping.
)
set "VPY=%ROOT%venv\Scripts\python.exe"

rem ---------- 3. dependencies ----------
echo.
echo [3/5] Installing dependencies (OpenVINO GenAI + PyAV)...
"%VPY%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
"%VPY%" -m pip install -i https://pypi.org/simple --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple openvino openvino-genai av numpy requests tqdm
if errorlevel 1 (
    echo   [ERROR] Dependency installation failed. Check your network.
    pause
    exit /b 1
)
echo   Dependencies ready.

rem ---------- 4. models ----------
echo.
echo [4/5] Downloading Whisper models (from ModelScope)...
"%VPY%" "%ROOT%tools\download_models.py"
if errorlevel 1 (
    echo   [WARN] Model download did not fully succeed, you can rerun this script later.
)

rem ---------- 5. self check ----------
echo.
echo [5/5] Running environment self-check...
"%VPY%" "%ROOT%tools\check_env.py"

echo.
echo  ============================================================
echo    Setup finished.
echo.
echo    How to use:
echo      1) Drag audio/video files onto whisper.bat
echo      2) Or:  whisper.bat "D:\media\a.mp3" -l zh -f srt
echo  ============================================================
echo.
pause
