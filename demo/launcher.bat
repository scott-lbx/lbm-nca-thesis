@echo off
REM launcher.bat - starts launcher.py (persistent bridge service on port 8090).
REM After this is running, open demo/index.html (file://) and click the
REM start button - no need to run run_demo.bat manually.
REM NOTE: kept ASCII-only on purpose - non-ASCII REM/echo text in this file
REM previously got corrupted under the Big5/950 console codepage and broke
REM batch parsing entirely (variables read as bare commands). See launcher.py
REM and the demo README for the Chinese-language explanation of what this does.
setlocal

REM -- resolve repo root and demo dir from this script's own location ---------
set "DEMO_DIR=%~dp0"
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "VENV_PYEXE=%REPO%\.venv\Scripts\python.exe"

if exist "%VENV_PYEXE%" (
  set "PYEXE=%VENV_PYEXE%"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [launcher.bat] no python found: neither %VENV_PYEXE% nor "python" on PATH
    pause
    exit /b 1
  )
  set "PYEXE=python"
)

start "NCA launcher" powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location '%DEMO_DIR%'; & '%PYEXE%' launcher.py"
endlocal
