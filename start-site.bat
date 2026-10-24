@echo off
rem Host the svrestyle GUI on 127.0.0.1:%PORT% and share it through a Cloudflare quick tunnel.
rem The tunnel's https://....trycloudflare.com address prints below; Ctrl+C stops the tunnel, closing the
rem "svrestyle server" window stops the site. The address is also written to %SVRESTYLE_CACHE%\tunnel.log.
rem Set PORT, SVRESTYLE_CACHE, SVRESTYLE_FONTS, SVRESTYLE_TCGLIVE or SVRESTYLE_PTCG_SIM before running to override the defaults below.
setlocal
cd /d "%~dp0"
if not defined PORT set PORT=8765
if not defined SVRESTYLE_CACHE set "SVRESTYLE_CACHE=%USERPROFILE%\svrestyle-work\cache"
if not defined SVRESTYLE_FONTS set "SVRESTYLE_FONTS=%USERPROFILE%\svrestyle-work\fonts"
if not defined SVRESTYLE_TCGLIVE if exist "E:\TCGLive_Extract" set "SVRESTYLE_TCGLIVE=E:\TCGLive_Extract"
rem "Send to ptcg-sim" writes custom sets into the simulator's checkout; its running server reloads them
if not defined SVRESTYLE_PTCG_SIM if exist "%USERPROFILE%\Downloads\vibecodepokemontcg2-main\client\src" set "SVRESTYLE_PTCG_SIM=%USERPROFILE%\Downloads\vibecodepokemontcg2-main"
rem "Publish to site" uploads a set to the hosted simulator's R2 bucket through rclone (its remote holds the keys)
if not defined CUSTOM_SETS_R2_REMOTE (
  where rclone >nul 2>nul && set "CUSTOM_SETS_R2_REMOTE=r2:ptcg-foil/custom-sets"
)
set "PYTHONPATH=%~dp0"

set CLOUDFLARED=cloudflared
where cloudflared >nul 2>nul || set "CLOUDFLARED=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not "%CLOUDFLARED%"=="cloudflared" if not exist "%CLOUDFLARED%" (
  echo cloudflared not found: install it with  winget install Cloudflare.cloudflared
  pause
  exit /b 1
)

netstat -ano | findstr /r /c:"127.0.0.1:%PORT% .*LISTENING" >nul
if errorlevel 1 (
  echo Starting the site on http://127.0.0.1:%PORT%/ ...
  start "svrestyle server" python -m svrestyle gui --no-browser --port %PORT%
  "%SystemRoot%\System32\timeout.exe" /t 3 /nobreak >nul
) else (
  echo Something already listens on port %PORT%: using it.
)

echo Opening the Cloudflare tunnel (look for the trycloudflare.com address below)...
rem the log keeps the address too, for when this window is out of sight
"%CLOUDFLARED%" tunnel --no-autoupdate --logfile "%SVRESTYLE_CACHE%\tunnel.log" --url http://127.0.0.1:%PORT%
endlocal
