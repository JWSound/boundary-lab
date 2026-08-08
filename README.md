# Boundary Lab

<img src="assets/mainwindow.png" alt="Boundary Lab main window" width="700">

Boundary Lab is a GUI-based multiphysics acoustic simulation tool for loudspeaker design. It generates or imports loudspeaker meshes, infers exterior BEM or coupled FEM-BEM-LEM solving from the configured physical system, and presents SPL, directivity, radiation impedance, spinorama-style curves, and 3D balloon plots in the desktop application. Ath is the bundled geometry-generator provider.

### [Follow the official development thread on DIYAudio](https://www.diyaudio.com/community/threads/boundary-lab.440847/)

## Features

- Waveguide design editor with one-click geometry generation through the bundled [Ath4](https://at-horns.eu/) generator
- 3D mesh viewport for generated geometry and imported `.msh` files
- Physical-system editor for exterior BEM and coupled FEM-BEM-LEM models
- Prescribed-velocity and linear electrodynamic transducer components
- Channel controls for level, polarity, delay, and HPF/LPF crossover shaping
- Live horizontal/vertical directivity, on-axis response, spinorama, and radiation-impedance plots
- Plot-image, polar-data, on-axis channel-data, and balloon-data export
- 3D balloon viewer built directly from Fibonacci-sphere solve samples
- Project save/load with readable, backward-compatible `.blab.json` files

While not required, if modeling in Autodesk Fusion, the [Fusion2Msh](https://github.com/JWSound/fusiontomsh) add-in is strongly recommended for quick imports of mesh files into Boundary Lab.

## Windows quick start

1. Double-click `01_install_update_boundary-lab.bat` in the repository folder.
2. Follow the guided prompts. The installer creates the Python environment and
   can optionally prepare the Julia BEAT Engine CPU and NVIDIA CUDA solvers.
3. If the installer adds Git, Python, or Julia, close it and run it again when
   instructed so Windows can refresh the available commands.
4. Double-click `02_start_boundary_lab.bat` to launch Boundary Lab.

Rerun the installer later to update or repair the application. For Linux,
manual installation, Wine/Ath setup, OpenCL runtimes, Julia solvers, CUDA, and
server setup, see [Installation and Setup](docs/Installation%20and%20Setup.md).

## Documentation

- [Installation and Setup](docs/Installation%20and%20Setup.md)
- [User Guide](docs/User%20Guide.md)
- [Physical System Model](docs/Physical%20System%20Model.md)
- [Coupled Solver](docs/Coupled%20Solver.md)
- [Boundary Lab Server](docs/Boundary%20Lab%20Server.md)
- [CUDA Server Docker Image](docs/Docker.md)
- [Model Assumptions](docs/Model%20Assumptions.md)
- [Inputs and Outputs](docs/Inputs%20and%20Outputs.md)
- [Advanced CLI workflow](docs/advanced/cli-workflow.md)
- [BEAT Engine Core](docs/advanced/beat-engine-core.md)
- [BEAT Engine CPU](docs/advanced/beat-engine-CPU.md)
- [BEAT Engine CUDA](docs/advanced/beat-engine-CUDA.md)
- [Forward Beam Shape plot](docs/advanced/forward-beam-shape.md)
