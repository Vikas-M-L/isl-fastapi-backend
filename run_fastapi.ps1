Param(
  [int]$Port = 5001,
  [string]$HostAddr = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

# Activate venv if present (.venv, venv1, or venv2)
if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
} elseif (Test-Path .\venv1\Scripts\Activate.ps1) {
  . .\venv1\Scripts\Activate.ps1
} elseif (Test-Path .\venv2\Scripts\Activate.ps1) {
  . .\venv2\Scripts\Activate.ps1
}

# Load .env if present (for FRONTEND_ORIGINS)
if (Test-Path .env) {
  Write-Host "Loading .env..."
  foreach ($line in Get-Content .env) {
    if ($line -match "^\s*#") { continue }
    if ($line.Trim().Length -eq 0) { continue }
    $kv = $line -split '=',2
    if ($kv.Length -eq 2) { [System.Environment]::SetEnvironmentVariable($kv[0].Trim(), $kv[1].Trim()) }
  }
}

Write-Host "Starting FastAPI on http://${HostAddr}:${Port}"
python -m uvicorn app_fastapi:app --host $HostAddr --port $Port