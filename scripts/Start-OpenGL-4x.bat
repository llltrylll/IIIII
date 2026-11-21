@echo off
setlocal
cd /d "%~dp0"
set "WGPU_BACKEND=gl"
echo Starting Material Painter with OpenGL (WGPU_BACKEND=gl)...
"%~dp0MaterialPainter.exe"
exit /b %errorlevel%
