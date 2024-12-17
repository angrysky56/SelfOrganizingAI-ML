# Run as Administrator
$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping WSL..." -ForegroundColor Yellow
wsl --shutdown

Write-Host "Stopping Docker services..." -ForegroundColor Yellow
Stop-Service -Name "com.docker.service" -Force
Stop-Service -Name "com.docker.backend" -Force

Write-Host "Cleaning up Docker directories..." -ForegroundColor Yellow
Remove-Item -Force -Recurse "$env:ProgramData\Docker"
Remove-Item -Force -Recurse "$env:LOCALAPPDATA\Docker"
Remove-Item -Force -Recurse "$env:APPDATA\Docker"
Remove-Item -Force -Recurse "$env:LOCALAPPDATA\Docker Desktop"

Write-Host "Creating WSL config for F: drive..." -ForegroundColor Yellow
$wslConfig = @"
[wsl2]
localhostForwarding=true
kernelCommandLine = systemd.unified_cgroup_hierarchy=1
guiApplications=true
pageReporting=false
debugConsole=true
network=true
nestedVirtualization=false
root=F:\\wsl\\Ubuntu-22.04
autoMemoryReclaim=dropcache
sparseVhd=true
"@

New-Item -Path "$env:USERPROFILE\.wslconfig" -Value $wslConfig -Force

Write-Host "Creating Docker config for F: drive..." -ForegroundColor Yellow
$dockerConfig = @"
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false
}
"@

New-Item -Path "$env:USERPROFILE\.docker" -ItemType Directory -Force
New-Item -Path "$env:USERPROFILE\.docker\daemon.json" -Value $dockerConfig -Force

Write-Host "Cleaning WSL data..." -ForegroundColor Yellow
wsl --unregister docker-desktop
wsl --unregister docker-desktop-data

Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host "Please reinstall Docker Desktop and point it to F: drive during installation." -ForegroundColor Yellow
