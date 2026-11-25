# Builds a release, stages it in dist/, and creates icon-bearing Windows launchers.
#   pwsh scripts/package.ps1            # build, stage, zip
#   pwsh scripts/package.ps1 -NoBuild   # stage and zip an existing build
param([switch]$NoBuild)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
if (-not $NoBuild) {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$stage = Join-Path $root "dist/Material Painter"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item target/release/material_painter.exe (Join-Path $stage "MaterialPainter.exe")
Copy-Item -Recurse assets (Join-Path $stage "assets")
Copy-Item LICENSE, THIRD_PARTY_NOTICES.html, README.md $stage

# Keep the explicit batch launchers for diagnostics and advanced use.
# The user-facing EXE launchers below have embedded icons and are relocatable.
Copy-Item scripts/Start-OpenGL-4x.bat, scripts/Start-Vulkan-4x.bat, scripts/OPENGL-VIEWPORT-TEST.txt $stage

# Convert the existing application icon to ICO for the launcher PE resources.
Add-Type -AssemblyName System.Drawing
if (-not ("NativeIconMethods" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NativeIconMethods
{
    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool DestroyIcon(IntPtr handle);
}
'@
}

$iconSource = Join-Path $stage "assets/ui/icon.png"
$iconPath = Join-Path $stage "MaterialPainter.ico"
$sourceImage = $null
$iconBitmap = $null
$graphics = $null
$iconHandle = [IntPtr]::Zero
try {
    $sourceImage = [System.Drawing.Image]::FromFile($iconSource)
    $iconBitmap = [System.Drawing.Bitmap]::new(256, 256, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($iconBitmap)
    $graphics.Clear([System.Drawing.Color]::Transparent)
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $graphics.DrawImage($sourceImage, 0, 0, 256, 256)

    $iconHandle = $iconBitmap.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($iconHandle)
    $iconStream = [System.IO.File]::Create($iconPath)
    try {
        $icon.Save($iconStream)
    }
    finally {
        $iconStream.Dispose()
    }
}
finally {
    if ($iconHandle -ne [IntPtr]::Zero) {
        [void][NativeIconMethods]::DestroyIcon($iconHandle)
    }
    if ($graphics) { $graphics.Dispose() }
    if ($iconBitmap) { $iconBitmap.Dispose() }
    if ($sourceImage) { $sourceImage.Dispose() }
}

$icoBytes = [System.IO.File]::ReadAllBytes($iconPath)
if ($icoBytes.Length -lt 100 -or $icoBytes[0] -ne 0 -or $icoBytes[1] -ne 0 -or $icoBytes[2] -ne 1 -or $icoBytes[3] -ne 0) {
    throw "MaterialPainter.ico was not generated as a valid Windows icon"
}

$launcherSource = Join-Path $PSScriptRoot "windows-launcher.cs"
if (-not (Test-Path $launcherSource)) { throw "Windows launcher source not found: $launcherSource" }

$cscCandidates = @(
    (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
    (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
)
$csc = $cscCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $csc) {
    $compiler = Get-Command csc.exe -ErrorAction SilentlyContinue
    if ($compiler) { $csc = $compiler.Source }
}
if (-not $csc) { throw "C# compiler (csc.exe) was not found; cannot build icon-bearing launchers" }

$commonCompilerArgs = @(
    "/nologo",
    "/target:winexe",
    "/platform:x64",
    "/optimize+",
    "/reference:System.Windows.Forms.dll",
    "/win32icon:$iconPath"
)

$openGlLauncher = Join-Path $stage "Material Painter.exe"
$openGlArgs = @("/out:$openGlLauncher") + $commonCompilerArgs + @($launcherSource)
& $csc @openGlArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $openGlLauncher)) {
    throw "Failed to compile the default OpenGL launcher"
}

$vulkanLauncher = Join-Path $stage "Material Painter (Vulkan).exe"
$vulkanArgs = @("/out:$vulkanLauncher", "/define:VULKAN") + $commonCompilerArgs + @($launcherSource)
& $csc @vulkanArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $vulkanLauncher)) {
    throw "Failed to compile the Vulkan launcher"
}

# The icon is embedded in each launcher; no external icon path is needed.
Remove-Item -Force $iconPath

$zip = Join-Path $root "dist/MaterialPainter-$version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Staged $stage"
Write-Host "Wrote $zip"
if ($env:GITHUB_OUTPUT) { "version=$version" | Out-File -Append -Encoding utf8 -FilePath $env:GITHUB_OUTPUT }
