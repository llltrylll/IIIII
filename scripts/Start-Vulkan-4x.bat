@echo off
setlocal
cd /d "%~dp0"
set "WGPU_BACKEND=vulkan"
echo Starting Material Painter with Vulkan (WGPU_BACKEND=vulkan)...
"%~dp0Material Painter.exe"
exit /b %errorlevel%
