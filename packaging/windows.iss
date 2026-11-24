#ifndef Payload
  #error Pass /DPayload=<absolute staged runtime directory>
#endif
#ifndef AppVersion
  #define AppVersion "0.5.1.dev0"
#endif
#ifndef Flavor
  #define Flavor "cpu"
#endif

[Setup]
AppId={{557D58CB-ED76-465A-A8DE-19C03801A536}
AppName=Boundary Lab
AppVersion={#AppVersion}
AppPublisher=Boundary Lab
AppPublisherURL=https://github.com/JWSound/boundary-lab
DefaultDirName={localappdata}\Programs\Boundary Lab
DefaultGroupName=Boundary Lab
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist
OutputBaseFilename=Boundary-Lab-{#AppVersion}-windows-x64-{#Flavor}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\assets\256.ico
UninstallDisplayIcon={app}\Boundary Lab.exe
LicenseFile=..\LICENSE
CloseApplications=yes
CloseApplicationsFilter=Boundary Lab.exe,python.exe,pythonw.exe,julia.exe
RestartApplications=no

ChangesAssociations=yes

[Registry]
Root: HKCU; Subkey: "Software\Classes\.blab"; ValueType: string; ValueData: "BoundaryLab.Project"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Classes\.blab\OpenWithProgids"; ValueType: string; ValueName: "BoundaryLab.Project"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\BoundaryLab.Project"; ValueType: string; ValueData: "Boundary Lab project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\BoundaryLab.Project\DefaultIcon"; ValueType: string; ValueData: "{app}\Boundary Lab.exe,0"
Root: HKCU; Subkey: "Software\Classes\BoundaryLab.Project\shell\open\command"; ValueType: string; ValueData: """{app}\Boundary Lab.exe"" ""%1"""

[Files]
Source: "{#Payload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Payload}\user-files\examples\*"; DestDir: "{userdocs}\Boundary Lab\examples"; Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs
Source: "{#Payload}\user-files\documentation\*"; DestDir: "{userdocs}\Boundary Lab\documentation"; Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs

[Dirs]
Name: "{userdocs}\Boundary Lab\runs"; Flags: uninsneveruninstall

[Icons]
Name: "{group}\Boundary Lab"; Filename: "{app}\Boundary Lab.exe"; WorkingDir: "{userdocs}\Boundary Lab"
Name: "{group}\Boundary Lab Files"; Filename: "{userdocs}\Boundary Lab"
Name: "{group}\Boundary Lab Guide"; Filename: "{userdocs}\Boundary Lab\documentation\Boundary Lab Guide.pdf"
#if Flavor == "cuda"
Name: "{group}\Check NVIDIA CUDA"; Filename: "{app}\Boundary Lab.exe"; Parameters: "--check-cuda"
#endif
Name: "{group}\Uninstall Boundary Lab"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Boundary Lab.exe"; Description: "Launch Boundary Lab"; Flags: nowait postinstall skipifsilent
