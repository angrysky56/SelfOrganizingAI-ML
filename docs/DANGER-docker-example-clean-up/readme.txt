Don't use these raw!!! You must modify them based on your PC paths and even then they may cause you some issues, AI written- worked for me to retrieve 35gb on my main drive from wherever Docker had hidden unremovable files somewhere in. So if you want to get rid of all your containers this will nuke them and all the hidden containers too so **CAUTION**!!! Also I think it installs Docker back on C- just delete it again and run this-

You will need the Docker.exe, open a powershell as admin in the folder with the .exe and run something like this to get it off your main drive:

Start-Process 'Docker Desktop Installer.exe' -Wait -ArgumentList 'install', '--accept-license', '--installation-dir=F:\wsl\DockerDesktop', '--wsl-default-data-root=F:\wsl\DockerData', '--always-run-service'

Go to settings-> Docker Desktop-> resources to hook WSL in.

Your .wslconfig should be here- main drive/Users/your user name, if it exists careful with the configs- you can break stuff:

"C:\Users\angry\.wslconfig"

Good Luck!

Ty and Claude.