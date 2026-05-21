# Ath Setup

Boundary Lab bundles Ath runtime files in:

```text
ath/ath.exe
ath/ath.cfg
```

The GUI prefers this bundled `ath.exe` when generating geometry.

## Gmsh Requirement

Ath requires the Gmsh application. Boundary Lab bundles Gmsh in:

```text
gmsh/gmsh-4.15.2-Windows64/gmsh.exe
```

The Gmsh executable path is configured in `ath/ath.cfg`:

```text
MeshCmd = "E:\AthGUI\boundary-lab\gmsh\gmsh-4.15.2-Windows64\gmsh.exe %f -"
```

Boundary Lab updates this value on GUI startup and immediately before running Ath.

## OutputRootDir

Ath uses `OutputRootDir` in `ath/ath.cfg` to decide where generated files are written. Boundary Lab updates this value on GUI startup and immediately before running Ath.

The managed output root is:

```text
runs/ath_output
```

The value written to `ath.cfg` is an absolute path, because Ath expects one.

## Generated Files

Ath creates a case folder under `runs/ath_output`. Boundary Lab then discovers:

- the generated STL
- the generated Gmsh `.msh`
- surface physical names

Boundary Lab also writes a cleaned solver mesh beside the generated `.msh`.

## Git Notes

`ath.exe` and `ath.cfg` are intentionally included in this repository. Generated outputs under `runs/` are ignored.
