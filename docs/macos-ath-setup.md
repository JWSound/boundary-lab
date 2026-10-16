# Optional Ath generation on macOS

feat/macos-installer
Boundary Lab's macOS distribution excludes both Ath and Wine.
Mesh import, project editing, and solving do not require either tool. Ath design
source remains available in projects even when generation is not configured.
The [macOS installer pipeline](macos-installer.md) excludes the repository's
`ath/` directory and any Wine distribution from its resource inventory. The Python
wheel includes only `src/blab`, not the repository's Ath executable. Windows
bundling and Linux command-line Wine discovery are unchanged.

## Install the external tools

1. Download Ath from its [author's website](https://at-horns.eu/), review its
   separate license, and extract it into a persistent folder you control.
   The public download is described as free for non-commercial use. Boundary
   Lab's GPLv3 license does not grant additional rights to Ath.
2. Install Wine using the instructions from the
   [macOS Wine package maintainer](https://github.com/Gcenx/macOS_Wine_builds).
   Use upstream installation instructions rather than disabling Gatekeeper.
   An Intel Wine build on Apple Silicon also requires Apple's Rosetta; follow
   macOS's installation prompt where applicable.
3. Open Boundary Lab and choose **Edit → Ath Setup…**. This entry is macOS-only.
4. Select the extracted Ath `.exe`. Click **Find Wine**, or browse/paste the
   executable path. A typical Wine installation uses
   `/Applications/Wine Stable.app/Contents/Resources/wine/bin/wine`.
   Select the executable inside the bundle, not the `.app` directory itself.
5. Click **Test configuration**. It checks Wine's version and asks Ath to generate
   a small geometry through the same `-b` batch interface used by Boundary Lab.
   This test executes the selected programs, but does not download or install
   either tool. First use may initialize Wine and take several minutes.
6. Click **Save**. Paths are machine preferences, not stored in shared projects.
   Use **Cancel** to discard path edits. Clear both paths and save to disconnect
   Ath. Moving or uninstalling either tool requires updating these paths.

The setup test checks geometry output, not meshing or numerical accuracy.
No specific external Ath/Wine/macOS combination has yet been qualified as part
of this setup feature. Run a real generation after setup to verify meshing.

## Files and troubleshooting

Wine uses a dedicated prefix at
`~/Library/Application Support/Boundary Lab/ath-wine`, outside the application.
The setup probe uses disposable scratch files; normal designs retain their
usual run directories. Boundary Lab never installs Wine or modifies its bundle.
The explicit Wine path works when launching from Finder without shell PATH setup.

If startup fails, check that Wine itself runs and that required Rosetta support
is installed. If Ath produces no geometry, check its version and support for the
`-b` batch interface. Diagnostic text is shown in the setup dialog. A failed
setup does not prevent imported meshes or existing generated meshes from being
used. The normal generation action reports **Ath setup required** when external
paths are absent or invalid.
