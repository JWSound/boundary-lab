# Noncubic cavity FEM fixtures

These Gmsh 4.1 fixtures describe the same rigid 470 x 330 x 220 mm rectangular
cavity with a nominal 14 mm tetrahedral element size.

| Fixture | Radiator patch | Intended first-order modal overlap |
|---|---|---|
| `noncubic.msh` | None | Analytic rigid-cavity reference |
| `noncubic_driven.msh` | Offset on the -Y wall | Broad X/Y/Z participation |
| `noncubic_driven_x_center.msh` | Centered on the -X wall | Excites X; suppresses Y and Z |
| `noncubic_driven_y_center.msh` | Centered on the -Y wall | Excites Y; suppresses X and Z |
| `noncubic_driven_z_center.msh` | Centered on the -Z wall | Excites Z; suppresses X and Y |

Each generated centered patch is 100 x 80 mm. The volume is physical group
`Body1` (tag 1), the patch is `Radiator` (tag 2), and all remaining surfaces
are `Body1_boundary` (tag 3).

The centered variants are generated from `noncubic_patch.geo`. From the
repository root:

```powershell
$gmsh = "gmsh/gmsh-4.15.2-Windows64/gmsh.exe"
& $gmsh tests/fixtures/noncubic_cavity/noncubic_patch.geo -3 -format msh41 `
  -o tests/fixtures/noncubic_cavity/noncubic_driven_x_center.msh `
  -setnumber PatchWall 0 -setnumber PatchU 0 -setnumber PatchV 0
& $gmsh tests/fixtures/noncubic_cavity/noncubic_patch.geo -3 -format msh41 `
  -o tests/fixtures/noncubic_cavity/noncubic_driven_y_center.msh `
  -setnumber PatchWall 1 -setnumber PatchU 0 -setnumber PatchV 0
& $gmsh tests/fixtures/noncubic_cavity/noncubic_patch.geo -3 -format msh41 `
  -o tests/fixtures/noncubic_cavity/noncubic_driven_z_center.msh `
  -setnumber PatchWall 2 -setnumber PatchU 0 -setnumber PatchV 0
```

`PatchU` and `PatchV` can be changed to create off-centre variants. Their axes
are Y/Z for an X wall, X/Z for a Y wall, and X/Y for a Z wall.

`TargetSize` controls the nominal tetrahedral element size. The convergence
fixtures use 20 mm (coarse), 14 mm (baseline), and 10 mm (fine). For example:

```powershell
& $gmsh tests/fixtures/noncubic_cavity/noncubic_patch.geo -3 -format msh41 `
  -o tests/fixtures/noncubic_cavity/noncubic_driven_x_center_h20.msh `
  -setnumber TargetSize 20 -setnumber PatchWall 0 `
  -setnumber PatchU 0 -setnumber PatchV 0
```

The unsuffixed centered fixtures are the 14 mm baseline. `_h20` and `_h10`
identify the coarse and fine variants respectively. Each resolution contains
X-, Y-, and Z-wall fixtures so source selectivity can be checked independently
of mesh density.

## Elements-per-wavelength diagnostic

The fixture-specific dispersion diagnostic samples exact axial, planar-oblique,
and fully oblique rigid-cavity modes from about 2 to 10 kHz. It evaluates their
Rayleigh quotients on the 20, 14, and 10 mm meshes and groups relative frequency
error by nominal elements per wavelength:

```powershell
julia --project=src/blab/solvers/julia_local `
  src/blab/solvers/julia_local/scripts/analyze_noncubic_ppw.jl
```

For these first-order tetrahedral meshes, the sampled results support
conservative working limits of about 17 elements per wavelength for 1% modal
frequency error, 12 for 2%, 8 for 5%, and 6 for 10%. These limits use the
requested uniform target size; production checks should use the actual local
edge-size and element-quality distributions.
