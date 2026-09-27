@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo ERROR: Project Python was not found at "%PYTHON%" 1>&2
  echo Run uv sync --locked --extra dev from the repository root. 1>&2
  exit /b 1
)

pushd "%PROJECT_ROOT%"
"%PYTHON%" -m fog.scripts.automated_training %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
