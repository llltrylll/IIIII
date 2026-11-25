; Inno Setup script for the Windows installer. Build the staged folder first
; (scripts/package.ps1), then:  iscc /DAppVersion=0.1.0 installer\material-painter.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6E4B7C2A-3F1D-4C8E-9A5B-2D7F1E0C9B34}
AppName=Material Painter
AppVersion={#AppVersion}
AppPublisher=Material Painter
AppPublisherURL=https://github.com/alla2001/material-painter
AppSupportURL=https://github.com/alla2001/material-painter/issues
DefaultDirName={autopf}\Material Painter
DefaultGroupName=Material Painter
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=MaterialPainter-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
UninstallDisplayIcon={app}\Material Painter.exe

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\Material Painter\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Material Painter"; Filename: "{app}\Material Painter.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Material Painter.exe"
Name: "{group}\Material Painter (Vulkan)"; Filename: "{app}\Material Painter (Vulkan).exe"; WorkingDir: "{app}"; IconFilename: "{app}\Material Painter (Vulkan).exe"
Name: "{group}\Uninstall Material Painter"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Material Painter"; Filename: "{app}\Material Painter.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Material Painter.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Material Painter.exe"; Description: "{cm:LaunchProgram,Material Painter}"; Flags: nowait postinstall skipifsilent
