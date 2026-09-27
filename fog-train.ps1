param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Project environment missing. Run uv sync --locked --extra dev from the repository root.'
}

Push-Location $ProjectRoot
try {
    & $Python -m fog.scripts.automated_training @CommandArgs
    $Result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $Result
