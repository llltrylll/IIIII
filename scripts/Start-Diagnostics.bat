@echo off
setlocal
cd /d "%~dp0"

rem Preserve the renderer selected by the user/system. This launcher only enables tracing.
set "RUST_BACKTRACE=1"
set "MP_STARTUP_TRACE=%~dp0Project-Open-Diagnostic.txt"
set "MP_STARTUP_FRAMES=3000"

if exist "%~dp0Project-Open-Diagnostic.txt" del /q "%~dp0Project-Open-Diagnostic.txt"
> "%~dp0Project-Open-Launcher.log" echo Material Painter project-open diagnostic launcher
>> "%~dp0Project-Open-Launcher.log" echo Started: %DATE% %TIME%
>> "%~dp0Project-Open-Launcher.log" echo Backend override: none
>> "%~dp0Project-Open-Launcher.log" echo Trace: %MP_STARTUP_TRACE%
>> "%~dp0Project-Open-Launcher.log" echo Executable: %~dp0MaterialPainter.exe

echo Starting Material Painter with project-open diagnostics.
echo The renderer is not changed. Open the Airfox project and wait for it to finish.
echo The report will be written beside this launcher as Project-Open-Diagnostic.txt.
echo.
"%~dp0MaterialPainter.exe" >> "%~dp0Project-Open-Launcher.log" 2>&1
set "APP_EXIT=%ERRORLEVEL%"

>> "%~dp0Project-Open-Launcher.log" echo Exit code: %APP_EXIT%
>> "%~dp0Project-Open-Launcher.log" echo Ended: %DATE% %TIME%

if not "%APP_EXIT%"=="0" (
  echo.
  echo Material Painter exited with code %APP_EXIT%.
  echo Check Project-Open-Diagnostic.txt and Project-Open-Launcher.log.
  echo.
  pause
)
exit /b %APP_EXIT%
