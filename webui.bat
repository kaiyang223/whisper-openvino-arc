@echo off
rem ============================================================
rem  Whisper local transcription - Web UI
rem
rem  Starts a local web server and opens your browser.
rem  Stop it with Ctrl+C in this window.
rem
rem  Usage:
rem     webui.bat                 normal start (auto opens browser)
rem     webui.bat --port 9000     use a custom port
rem     webui.bat --no-browser    do not open the browser
rem
rem  NOTE: this file is intentionally ASCII-only with CRLF endings.
rem ============================================================

setlocal
chcp 65001 >nul
title Whisper Web UI - OpenVINO on Intel Arc

set "ROOT=%~dp0"
set "PY=%ROOT%venv\Scripts\python.exe"

if not exist "%PY%" goto :nopython

"%PY%" "%ROOT%webui\server.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo   [exit code %RC%]
    pause
)
exit /b %RC%

:nopython
echo.
echo   [ERROR] Python environment not found:
echo           %PY%
echo.
echo   Please run setup.bat first (right-click, "Run as administrator").
echo.
pause
exit /b 1
