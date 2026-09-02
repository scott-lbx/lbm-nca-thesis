@echo off
REM ============================================================================
REM  run_gh_backend.bat - starts the inference backend for Grasshopper use
REM  ONLY (no web page, no http.server, no browser).
REM
REM  GH sends geometry straight to the WebSocket server that lives inside
REM  the backend process - it never touches demo/index.html or the
REM  http.server, so this script starts exactly one thing:
REM    1. Independent window: python -m lbmnca.bridge.main
REM       (WebSocket server on ws://localhost:8765 + the Taichi GGUI window,
REM        which is the ONLY place the flow field is drawn - GH itself never
REM        shows the flow field, only a status/ack string)
REM
REM  Portable: REPO is derived from this file's own location, not hardcoded
REM  to a user name or drive letter, so the same script works on any machine
REM  that has this git repo checked out (demo/ sits directly under the repo
REM  root, alongside src/, configs/, checkpoints/).
REM
REM  To close: close the "NCA main.py (GH)" console window.
REM
REM  NOTE: kept ASCII-only on purpose - non-ASCII REM/echo text in batch files
REM  previously got corrupted under the Big5/950 console codepage and broke
REM  parsing entirely (variables read back as bare, unrecognized commands).
REM  See demo/run_demo.bat for the same note.
REM ============================================================================

setlocal

REM -- resolve repo root from this script's own location (demo\) --------------
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "VENV_PYEXE=%REPO%\.venv\Scripts\python.exe"

if exist "%VENV_PYEXE%" (
  set "PYEXE=%VENV_PYEXE%"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [run_gh_backend] no python found: neither %VENV_PYEXE% nor "python" on PATH
    pause
    exit /b 1
  )
  set "PYEXE=python"
)

if not exist "%REPO%\src\lbmnca\bridge\main.py" (
  echo [run_gh_backend] lbmnca.bridge.main not found under: %REPO%\src
  echo [run_gh_backend] check this script's location - it must stay at
  echo [run_gh_backend] ^<repo root^>\demo\run_gh_backend.bat
  pause
  exit /b 1
)

echo [run_gh_backend] repo = %REPO%
echo [run_gh_backend] starting lbmnca.bridge.main (own window)...
start "NCA main.py (GH)" powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location '%REPO%'; & '%PYEXE%' -m lbmnca.bridge.main"

echo.
echo [run_gh_backend] waiting 20 seconds for model load...
REM ping-based delay instead of `timeout /t` - timeout needs a real console
REM input handle and errors out ("Input redirection is not supported") when
REM launched from a non-interactive shell (e.g. a scheduler or automation);
REM ping has no such dependency and 21 pings at 1s each ~= 20s.
ping -n 21 127.0.0.1 >nul

echo [run_gh_backend] check the "NCA main.py (GH)" window: it should show
echo [run_gh_backend]   [main] WS server ws://localhost:8765
echo [run_gh_backend] and a separate Taichi GGUI window should have opened
echo [run_gh_backend] (640x360, the only place the flow field is drawn).
echo [run_gh_backend] If it is still loading, wait and check again - the
echo [run_gh_backend] model load takes ~20-40 seconds on first start.
echo.
echo [run_gh_backend] Once ready: in Rhino/Grasshopper, set the "run" toggle
echo [run_gh_backend] to True on the analyze_component_standalone.py component.
echo [run_gh_backend] See demo/README.md section "Grasshopper" for the
echo [run_gh_backend] component Input/Output setup.
endlocal
