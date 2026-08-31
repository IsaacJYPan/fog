param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

$Candidates = @()
if (Test-Path -LiteralPath $VenvPython) {
    $Candidates += [pscustomobject]@{ Python = $VenvPython; PythonPath = $null; Label = 'project venv' }
}
$SystemPython = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $SystemPython) {
    $Candidates += [pscustomobject]@{ Python = $SystemPython.Source; PythonPath = $null; Label = 'system Python' }
}

$Selected = $null
foreach ($Candidate in $Candidates) {
    $PreviousPythonPath = $env:PYTHONPATH
    if ($Candidate.PythonPath) {
        $env:PYTHONPATH = $Candidate.PythonPath
    } else {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    $CandidateWorks = $false
    try {
        & $Candidate.Python -c 'import torch, yaml; assert torch.cuda.is_available()' 2>$null
        $CandidateWorks = ($LASTEXITCODE -eq 0)
    } catch {
        $CandidateWorks = $false
    }
    if ($CandidateWorks) {
        $Selected = $Candidate
        break
    }
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
}

if ($null -eq $Selected) {
    throw @"
No working Python environment with PyTorch, PyYAML, and CUDA was found.
The existing .venv points to a Python installation that no longer exists.
Create a fresh environment with your current Python, then install this project:
  python -m venv .venv-new
  .\.venv-new\Scripts\python.exe -m pip install -e .
"@
}

if ($Selected.PythonPath) {
    $env:PYTHONPATH = $Selected.PythonPath
}

Write-Host "Using $($Selected.Label): $($Selected.Python)"
Push-Location $ProjectRoot
try {
    & $Selected.Python -m fog.scripts.automated_training @CommandArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
