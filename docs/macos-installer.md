# macOS Apple Silicon candidate

The macOS build produces a standalone `Boundary Lab.app` in a drag-to-Applications
DMG. The initial candidate targets macOS 14+ on Apple Silicon and bundles CPU
solving and the remote-server client. It does not bundle CUDA, Metal, Ath, or Wine.
Ath generation is optional and uses [external tool setup](macos-ath-setup.md).
The built-in parametric generator and imported meshes do not need Ath.

This is a development/evaluation candidate, not a stable public release. The
application retains its current development version and exact Git-pinned BEAT
revision. A stable release must select version metadata and adopt a qualified,
published engine wheel with a matching checksum under the normal release process.
No engine version is silently substituted by these scripts.

## Build and sign

Use native ARM64 Python 3.12+, Xcode command-line tools, and a Developer ID
Application certificate with its private key in Keychain. No Developer ID
Installer certificate is needed for a DMG.

```sh
.venv/bin/python scripts/build_macos_runtime.py --output build/macos-candidate
.venv/bin/python scripts/build_macos_installer.py \
  --app 'build/macos-candidate/Boundary Lab.app' \
  --output dist/macos-signed --identity YOUR_CERTIFICATE_SHA1
.venv/bin/python scripts/test_macos_installer.py \
  --dmg dist/macos-signed/Boundary-Lab-0.5.0.dev0-mac-arm64.dmg \
  --output 'build/Mac Installed Qualification'
```

Outputs must be new directories. The build downloads checksum-pinned standalone
Python and Julia, uses pinned native Python wheels, and stages Julia packages in
an isolated build depot. The resource whitelist omits `ath/` and Wine. The app
contains a native launcher, icons, bundled plugins, examples, and documentation.

Signing copies the staged app, verifies its resource inventory, signs native
executables/libraries and nested frameworks/apps from the inside out, inventories the signed
bytes, and finally seals the outer app. Julia receives JIT/library-loading
entitlements needed by its generated package caches. The staged app is unchanged.
Do not mutate a signed app or its manifest. Rebuild into a new directory instead.

Qualification mounts and copies the DMG into a spaced path, verifies signatures,
checks the native launcher, and runs isolated GUI/VTK rendering, native Gmsh
meshing, and validated exterior/coupled CPU solves with blocked HTTP(S) proxies.
It repeats with the same user cache, checks complex results/provenance, and
verifies that installed resource hashes have not changed. It does not qualify
external Ath/Wine or GPU solving.

## Notarize after qualification

Submit the complete signed DMG using an existing Keychain notary profile:

```sh
.venv/bin/python scripts/notarize_macos_installer.py \
  --dmg dist/macos-signed/Boundary-Lab-0.5.0.dev0-mac-arm64.dmg \
  --qualification 'build/Mac Installed Qualification' --profile YOUR_NOTARY_PROFILE
```

The command checks the qualified DMG hash before upload and retains a resumable
submission receipt. After
Apple accepts it, staple and validate the ticket, assess the DMG and installed
app with Gatekeeper, and generate the final checksum after stapling. The adjacent
`SHA256SUMS-unstapled.txt` is only a build-stage checksum, not the final download
checksum. Do not publish a failed or unqualified candidate.

## User files and distribution

Examples and docs are seeded without overwriting edits into `~/Documents/Boundary
Lab`. Logs, Matplotlib state, and version-specific Julia caches live outside the
app in the user's Library. Native launcher diagnostics are in
`~/Library/Logs/Boundary Lab/launcher.log`. macOS scratch paths use private system
temporary directories without spaces to avoid Julia Downloads precompile errors.

Before public distribution, complete the exact dependency license/source delivery
inventory (including Qt/Gmsh and their bundled libraries), qualification of a
quarantined download on a clean Mac, and upgrade testing. Notarization does not
replace these checks. No signing credentials are committed or exposed to PR CI.
