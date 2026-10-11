@echo off
setlocal
cd /d "%~dp0"
echo ===========================================
echo   VoxelPool Explorer - first-time setup
echo ===========================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" was not found. Install Python 3.10+ and enable PATH.
  pause
  exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js was not found. Install Node.js 18+ and reopen this window.
  pause
  exit /b 1
)
if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -m venv backend\.venv
)
echo Installing backend dependencies...
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo Backend dependency installation failed.
  pause
  exit /b 1
)
echo Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
  echo Frontend dependency installation failed.
  pause
  exit /b 1
)
cd ..
start "VoxelPool API" cmd /k "cd /d ""%~dp0backend"" && ""%~dp0backend\.venv\Scripts\python.exe"" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "VoxelPool Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
timeout /t 3 /nobreak >nul
start "" http://localhost:5173
echo.
echo Both services have been launched in separate terminals.
echo Dashboard: http://localhost:5173
echo API docs:  http://127.0.0.1:8000/docs
echo Keep both terminal windows open while using the app.
pause
