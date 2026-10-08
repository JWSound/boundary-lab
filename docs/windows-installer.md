# Windows x64 installer

Boundary Lab's desktop bundle contains private Python and Julia runtimes, BEAT
Engine, Qt/PyVista/VTK, Gmsh, and Ath. Users do not need Python, Julia, Git, pip,
or a CUDA toolkit installation. The initial target is Windows 10 1809+ / Windows
11 x64, with the OS-provided .NET Framework used by the small desktop launcher.

## User files

The per-user installer defaults to `%LOCALAPPDATA%\Programs\Boundary Lab`.
The Windows Documents Known Folder (including OneDrive or redirected Documents)
contains:

```text
Documents\Boundary Lab\
  examples\
  runs\                 generated meshes, configurations, and geometry
  documentation\        PDF guide and repository documentation
```

Existing example and documentation files are never overwritten. New files are
added on installation or launch. To obtain a fresh copy of an edited example,
rename the old file and launch again. Uninstall preserves this entire directory.
Projects can still be saved anywhere. Source-checkout workflows retain their
existing paths; these defaults apply to packaged launches.

Settings retain the application's existing per-user Qt settings location. Logs,
temporary solver files, Matplotlib caches, and version-specific Julia compilation
caches live in `%LOCALAPPDATA%\Boundary Lab`. No compilation or solve writes to
the installation directory. First use compiles Julia code locally and can take
several minutes; it requires no dependency downloads.

## CPU and NVIDIA packages

The `cpu` installer includes everything needed for CPU solving and exposes CPU
and remote-server choices. Saved preferences for an unbundled GPU backend fall
back to CPU.

The `cuda` installer includes CPU fallback plus the pinned NVIDIA toolkit,
compiler, BLAS, sparse, and solver libraries. A compatible NVIDIA GPU and driver
must already be present. Driver installation is managed separately by NVIDIA;
Boundary Lab does not modify drivers. Use **Check NVIDIA CUDA** in the Start Menu
after installing/updating the driver, then select CUDA in Preferences. The check
uses the actual bundled Julia toolkit, writes `logs\cuda-check.log`, and does not
download packages. CPU remains the initial GUI default. AMD ROCm is not bundled.

## Building a candidate

Use Windows x64, Python 3.13 x64, Git, the .NET Framework C# compiler, and Inno
Setup 6. Build-time networking is needed; installed operation is offline.

```powershell
python scripts/build_windows_runtime.py --backend cpu
python scripts/verify_windows_runtime.py --payload build/windows-cpu --destination "build/Relocated Boundary Lab" --solve
python scripts/build_windows_installer.py --payload build/windows-cpu --iscc "C:/Program Files (x86)/Inno Setup 6/ISCC.exe"
```

For the combined package, use `--backend cuda` and the `windows-cuda` payload.
Qualify it with both `--backend cpu --solve` and `--backend cuda --solve`; the
latter requires compatible NVIDIA hardware. The manual GitHub Actions workflow
builds an unsigned candidate and qualifies CPU operation. Hosted CPU-only runners
cannot qualify CUDA numerical execution.

Runtime archives and BEAT are checksum-pinned in `packaging/runtime-lock.json`.
The complete Python dependency closure is version-pinned in
`packaging/requirements-win.txt`. Builds download wheels, stage Julia packages
and lazy artifacts from a fresh build depot, retain upstream license files, and
record each wheel and installed file hash. They never copy a developer virtual
environment or user Julia depot. Runtime outputs must be new directories.

For application-only iteration, `scripts/restage_windows_application.py --base
build/windows-cpu --output build/windows-cpu-next` copies verified dependency
files to a new directory and rebuilds the application. It rejects changed runtime
or dependency pins. Qualify the new payload again before compiling its installer.

After compiling, `scripts/test_windows_installer.py --installer dist/<candidate>.exe
--directory "build/Installed Evaluation" --backend cpu` performs a guarded silent
installation, launches its initializer, qualifies installed solves, and uninstalls.
It refuses an existing Boundary Lab installation and preserves Documents files.

The verifier uses the bundled Python and Julia, redirected test user directories,
invalid developer runtime overrides, and blocked conventional HTTP(S) proxies.
It renders the GUI and a VTK scene, generates an Ath/Gmsh mesh, validates before
solving exterior and coupled versions of the sealed example at 500 Hz, checks
finite complex results and engine provenance, and checks installation hashes.
These are packaging checks, not mesh-convergence or acoustic-accuracy claims.

`dist/` contains the installer and SHA-256 file. This build path creates **unsigned
evaluation candidates** and never publishes them. Before public release, qualify
on clean CPU-only Windows hardware with networking disabled, test upgrades and
uninstall, configure production signing, and complete redistribution notices and
corresponding-source delivery for the exact dependency set. Retain Ath's existing
redistribution permission. A development-machine offline smoke run is not a
substitute for clean-machine qualification.

## Local evaluation — 2026-10-07

Both unsigned `0.5.0.dev0` candidates were built and tested on this Windows
development machine. The runtime pins are Python 3.13.13, Julia 1.12.6, and
BEAT Engine 0.4.0. CUDA testing used an NVIDIA RTX 2080 Ti with driver 596.49.

| Candidate | Installer bytes | Runtime ID |
| --- | ---: | --- |
| CPU | 500,937,964 | `win-x64-51228b8e95e78446` |
| CPU + CUDA | 1,649,648,751 | `win-x64-74abd565404b0487` |

Installers and adjacent SHA-256 files are in `dist/`. Both passed silent per-user
installation into fresh directories containing spaces, launcher initialization,
GUI and VTK rendering, Ath/Gmsh generation using the desktop `pythonw.exe` worker,
and offline exterior and coupled solves at 500 Hz. Generated files and solve
outputs used a redirected test Documents path containing spaces and accented
characters. Results were finite and retained complex quantities and engine
provenance. All inventoried installed files remained unchanged.

Both registered uninstallers removed the application executable and uninstall
registration while preserving the real Documents examples and PDF guide. The
test installations have been removed. Detailed evidence is retained under
`build/Installed CPU Evaluation-lifecycle.json`,
`build/Installed CUDA Evaluation-lifecycle.json`, and `build/qualification-*`.

Focused regression runs passed (84 path/GUI/headless checks, 59 runtime/Ath
checks, and four final runtime checks; these suites overlap). Ruff checks passed.
These are local evaluation results, not certification on a clean Windows machine.
Public-release signing, clean CPU-only hardware testing, and upgrade qualification
remain release gates.

### Console-window suppression follow-up

[Application PR #38](https://github.com/JWSound/boundary-lab/pull/38) and
[BEAT Engine PR #61](https://github.com/JWSound/BEAT_Engine/pull/61) suppress Windows console windows
for Ath, its Gmsh worker, cancellation's `taskkill`, the CUDA probe, and Julia
solves. These fixes preserve Ath's process-group flag and the existing output
pipes. The engine's Windows transport test verifies that the child has no
console and still streams results across reused submissions.

The candidates listed above predate this fix. The published BEAT Engine 0.4.0
wheel remains pinned and has not been modified. Before shipping the fix, publish
a new engine version, update the application version check and both dependency
pins (including the wheel hash), then rebuild and qualify both installer editions.
