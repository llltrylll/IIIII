# Builds a release and stages it in dist/, then zips it.
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

# The backend is selected before wgpu creates its instance. Ship explicit
# launchers so users do not accidentally start the default Vulkan backend
# when they intend to test the OpenGL viewport.
Copy-Item scripts/Start-OpenGL-4x.bat, scripts/Start-Vulkan-4x.bat, scripts/Start-Diagnostics.bat, scripts/OPENGL-VIEWPORT-TEST.txt $stage

$zip = Join-Path $root "dist/MaterialPainter-$version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip
Write-Host "Staged $stage"
Write-Host "Wrote $zip"
if ($env:GITHUB_OUTPUT) { "version=$version" | Out-File -Append -Encoding utf8 -FilePath $env:GITHUB_OUTPUT }
