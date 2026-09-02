@echo off
REM ============================================================================
REM  run_demo.bat - NCA demo one-click launcher
REM
REM  Does three things in order:
REM    1. Independent window: python -m lbmnca.bridge.main (WebSocket server +
REM       Taichi GGUI window)
REM    2. Independent window: python -m http.server 8080 at the repo root
REM       (root, because demo/index.html loads ./app.js and the server root
REM        must cover the demo/ directory)
REM    3. Open the browser to the demo page once servers are up
REM
REM  Per workspace CLAUDE.md rule "long-running commands always get their own
REM  window", both services use `start` to open a separate window.
REM  To close the demo: just close those two PowerShell/console windows.
REM
REM  NOTE: kept ASCII-only on purpose - non-ASCII REM/echo text in batch files
REM  previously got corrupted under the Big5/950 console codepage and broke
REM  parsing entirely (variables read back as bare, unrecognized commands).
REM  See demo/README.md for the Chinese-language explanation.
REM ============================================================================

setlocal

REM -- resolve repo root from this script's own location (demo\) -------------
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "VENV_PYEXE=%REPO%\.venv\Scripts\python.exe"
set "HTTP_PORT=8080"
set "DEMO_URL=http://localhost:%HTTP_PORT%/demo/index.html"

if exist "%VENV_PYEXE%" (
  set "PYEXE=%VENV_PYEXE%"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [run_demo] no python found: neither %VENV_PYEXE% nor "python" on PATH
    pause
    exit /b 1
  )
  set "PYEXE=python"
)

if not exist "%REPO%\src\lbmnca\bridge\main.py" (
  echo [run_demo] lbmnca.bridge.main not found under: %REPO%\src
  echo [run_demo] check this script's location - it must stay at
  echo [run_demo] ^<repo root^>\demo\run_demo.bat
  pause
  exit /b 1
)

REM -- 1) inference service: own window, -NoExit keeps log visible ------------
echo [run_demo] starting lbmnca.bridge.main (own window)...
start "NCA main.py" powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location '%REPO%'; & '%PYEXE%' -m lbmnca.bridge.main"

REM -- 2) static file server: repo root, covers demo/ --------------------------
echo [run_demo] starting http.server %HTTP_PORT% (own window, root=%REPO%)...
start "NCA http.server" powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location '%REPO%'; & '%PYEXE%' -m http.server %HTTP_PORT%"

REM -- 3) wait for services (model load is slow; page can open first,
REM       click Connect once it's ready) --
echo [run_demo] waiting 8 seconds for services to come up...
timeout /t 8 /nobreak >nul

echo [run_demo] opening %DEMO_URL%
start "" "%DEMO_URL%"

echo.
echo [run_demo] done. On the page: click Connect (light turns green) then draw a building then Send.
echo [run_demo] if Connect fails, the backend is still loading the model - wait and click again.
endlocal
