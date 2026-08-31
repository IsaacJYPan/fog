@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo ERROR: Project Python was not found at "%PYTHON%" 1>&2
  echo Recreate the environment or run fog-train.ps1 with an allowed execution policy. 1>&2
  exit /b 1
)

"%PYTHON%" -c "import torch, yaml; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
  echo ERROR: The project environment cannot import PyTorch and PyYAML with CUDA enabled. 1>&2
  echo Run: "%PYTHON%" -m fog.scripts.automated_training doctor --job configs\automated_gen9ou_typed_league.yaml 1>&2
  exit /b 1
)


pushd "%PROJECT_ROOT%"
"%PYTHON%" -m fog.scripts.automated_training %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
