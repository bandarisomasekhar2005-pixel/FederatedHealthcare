@echo off
cd /d "%~dp0"
start "Federated Healthcare Backend" cmd /k start_backend.bat
timeout /t 3 /nobreak >nul
start "Federated Healthcare Frontend" cmd /k start_frontend.bat
timeout /t 5 /nobreak >nul
start "" http://localhost:5173
echo.
echo Backend:  http://localhost:8000
echo API docs: http://localhost:8000/docs
echo Frontend: http://localhost:5173
