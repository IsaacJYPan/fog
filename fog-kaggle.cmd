@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m fog.scripts.kaggle_results %*
exit /b %errorlevel%
