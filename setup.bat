@echo off
setlocal
cd /d "%~dp0"
echo [1/4] Creating Python virtual environment...
if not exist ".venv\Scripts\python.exe" py -m venv .venv
echo [2/4] Installing Python dependencies...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
echo [3/4] Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo [4/4] Initializing database...
call .venv\Scripts\python.exe -m backend.app.services.seed_db
echo.
echo Setup complete.
echo Run start_all.bat to start the application.
endlocal
