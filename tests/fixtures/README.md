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
- `SAWMOD/` contains the full and symmetry-reduced mesh variants used by
  interface-conformance, component-symmetry, and system-editor tests. The
  ready-to-open project lives in `examples/Multi_region_SAWMOD/`.

Ordinary cleaning and solver output belong under the ignored `runs/` tree.
