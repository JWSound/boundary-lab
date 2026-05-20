# Boundary Lab

Boundary Lab is a GUI-first Boundary Element Method tool for loudspeaker design. It uses Ath to generate loudspeaker surface meshes, runs BEM solves with `bempp-cl`, and shows SPL, directivity, radiation impedance, spinorama-style curves, and 3D balloon plots inside the desktop app.

## Features

- Ath `.cfg` editor with one-click geometry generation
- Mesh preview for generated Ath meshes and imported `.msh` files
- Multi-mesh and multi-radiator BEM solves
- Source controls for level, polarity, delay, and HPF/LPF crossover shaping
- Live horizontal/vertical directivity, on-axis response, spinorama, and impedance plots
- Optional 3D balloon plot viewer with spherical sampling
- Project save/load with readable `.blab.json` files

## Requirements

- Python 3.11 or newer
- Gmsh installed locally and available at the path referenced in `ath/ath.cfg`
- An OpenCL runtime for `bempp-cl`/`pyopencl`
- The bundled Ath runtime files in `ath/`

On Windows, the Intel CPU OpenCL runtime is a practical option even on many non-Intel systems.

## Install

From the repository root:

```bash
pip install -e ".[gui]"
```

## Run The GUI

```bash
blab gui
```

On startup, Boundary Lab updates `ath/ath.cfg` so Ath writes generated files into:

```text
runs/ath_output
```

Generated runs, solver outputs, and local project files are ignored by git.

## First Workflow

1. Launch the GUI with `blab gui`.
2. Write or import an Ath `.cfg` in the editor.
3. Click `Generate` to run Ath and load the generated mesh.
4. Open `Mesh Config` to enable/disable meshes or apply XYZ offsets.
5. Open `Source Config` to choose driven surfaces and source settings.
6. Set the frequency range and count.
7. Click `Solve`.
8. Use `View > Balloon Plot` after a solve if spherical sampling was enabled in Preferences.
9. Use `File > Save Project` to save editor, mesh, and source setup.

## Documentation

- [GUI user guide](docs/user-guide.md)
- [Ath setup](docs/ath-setup.md)
- [Project files](docs/project-files.md)
- [Advanced CLI workflow](docs/advanced/cli-workflow.md)
- [Advanced solver configuration](docs/advanced/solver-configuration.md)
- [Advanced examples](docs/advanced/examples.md)

## Notes

Boundary Lab uses prescribed radiator velocity drives. It does not currently model electro-mechanical driver behavior from T/S parameters, motor force factor, suspension compliance, or passive crossover networks.
