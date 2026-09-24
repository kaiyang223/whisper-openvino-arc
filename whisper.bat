@echo off
rem ============================================================
rem  Whisper local transcription - OpenVINO on Intel Arc
rem
rem  Usage:
rem    - Drag audio/video files onto this .bat file
rem    - or run from a command prompt:
rem        whisper.bat "D:\media\meeting.mp3"
rem        whisper.bat "D:\media" -r -l zh -f srt
rem        whisper.bat "D:\media\a.wav" --bench
rem
rem  NOTE: this file is intentionally ASCII-only with CRLF endings.
rem        All Chinese messages are printed by the Python program.
rem ============================================================

setlocal
chcp 65001 >nul
title Whisper - OpenVINO on Intel Arc

set "ROOT=%~dp0"
set "PY=%ROOT%venv\Scripts\python.exe"

if not exist "%PY%" goto :nopython

if "%~1"=="" goto :welcome

"%PY%" "%ROOT%transcribe.py" %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:welcome
echo.
"%PY%" "%ROOT%transcribe.py"
echo.
echo   (Drag audio or video files onto whisper.bat to transcribe them.)
echo.
pause
exit /b 0

:nopython
echo.
echo   [ERROR] Python environment not found:
echo           %PY%
echo.
echo   Please run setup.bat first (right-click, "Run as administrator").
echo.
pause
exit /b 1
