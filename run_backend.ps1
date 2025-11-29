Param(
  [string]$BindAddr="0.0.0.0",
  [int]$BindPort=5000
)

if (Test-Path .env) {
  Write-Host "Loading environment from .env"
  Get-Content .env | ForEach-Object {
    if ($_ -match '^(\s*#|\s*$)') { return }
    $name, $value = $_.Split('=',2)
    Set-Item -Path "Env:$name" -Value $value
  }
}

if (Test-Path .\app.py) {
  $env:FLASK_APP = "app.py"
  $msg = "Starting Flask backend on http://{0}:{1}" -f $BindAddr, $BindPort
  Write-Host $msg -ForegroundColor Green
  python app.py
} else {
  Write-Host "Error: Flask entrypoint 'backend/app.py' not found." -ForegroundColor Red
  Write-Host "Tip: Start the FastAPI detection server instead:" -ForegroundColor Yellow
  Write-Host "  .\run_fastapi.ps1 -Port 5001" -ForegroundColor Yellow
  exit 1
}
