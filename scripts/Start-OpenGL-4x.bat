@echo off
setlocal
cd /d "%~dp0"
set "WGPU_BACKEND=gl"
set "RUST_BACKTRACE=1"
set "MP_STARTUP_TRACE=%~dp0OpenGL-startup-trace.txt"

> "%~dp0OpenGL-launch.log" echo Material Painter OpenGL launcher
>> "%~dp0OpenGL-launch.log" echo Requested backend: %WGPU_BACKEND%
>> "%~dp0OpenGL-launch.log" echo Started: %DATE% %TIME%
>> "%~dp0OpenGL-launch.log" echo Executable: %~dp0MaterialPainter.exe

echo Starting Material Painter with OpenGL (WGPU_BACKEND=gl)...
echo If startup fails, this window will remain open.
"%~dp0MaterialPainter.exe" >> "%~dp0OpenGL-launch.log" 2>&1
set "APP_EXIT=%ERRORLEVEL%"

>> "%~dp0OpenGL-launch.log" echo Exit code: %APP_EXIT%
>> "%~dp0OpenGL-launch.log" echo Ended: %DATE% %TIME%

if not "%APP_EXIT%"=="0" (
  echo.
  echo Material Painter failed to start. Exit code: %APP_EXIT%
  echo Diagnostic files:
  echo   %~dp0OpenGL-launch.log
  echo   %APPDATA%\MaterialPainter\startup-error.log
  echo   %APPDATA%\MaterialPainter\crash.log
  echo.
  pause
)

exit /b %APP_EXIT%
