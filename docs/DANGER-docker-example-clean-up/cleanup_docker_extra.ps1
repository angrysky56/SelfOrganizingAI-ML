# Additional cleanup for remaining Docker files
$ErrorActionPreference = "SilentlyContinue"

Write-Host "Removing additional Docker folders..." -ForegroundColor Yellow
Remove-Item -Force -Recurse "C:\ProgramData\DockerDesktop"
Remove-Item -Force -Recurse "C:\Program Files\Docker"
Remove-Item -Force -Recurse "C:\Program Files\Docker Desktop"

Write-Host "Cleaning Windows temp files..." -ForegroundColor Yellow
Remove-Item -Force -Recurse "C:\Windows\Temp\*"
Remove-Item -Force -Recurse "$env:TEMP\*"

Write-Host "Additional cleanup complete!" -ForegroundColor Green