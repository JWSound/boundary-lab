# Test fixture inventory

The mesh files in this directory exercise geometry and solver cases that are
too large or specialized to construct inside individual tests.

- The root-level `femvolume`, `exterior`, and `exterior_conforming` meshes form
  the small baseline FEM/BEM interface case.
- The root-level `curvedinterface*` meshes exercise curved-interface matching
  and FEM mesh-convergence checks.
- `nonplanar_multisurface_interface/` covers one nonplanar interface split
  across multiple physical surface groups.
- `noncubic_cavity/` contains the documented analytic cavity and mesh-density
  convergence family.
- `SAWMOD/` contains full and symmetry-reduced interface/component cases. Its
  project file uses paths relative to that directory; the conformed exterior
  mesh is committed so the project opens ready to compile on a fresh clone.
- `SAWMOD_Full/` retains the separate full-domain SAWMOD geometry pair.
- `SKRAM/` is the multi-chamber, multi-interface compilation and symmetry case.

Project fixtures must not contain workstation-specific absolute paths. Keep a
derived mesh only when it is required to make a project immediately usable;
ordinary cleaning and solver output belong under the ignored `runs/` tree.
