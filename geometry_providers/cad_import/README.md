# CAD Import

A bundled Generator Plugin that turns a STEP file into a conformal FEM/BEM mesh
assembly. Solids become **air bodies** (bounded regions); surface bodies become
**exterior surface bodies**, which together form the BEM exterior. Meshing runs
the same headless pipeline as `blab cad-import`, so both produce identical
meshes for identical inputs.

## Use

The plugin is enabled once when introduced. Select **CAD Import** as the default
in **Edit > Generator Plugins...** and add a design.

1. **Open STEP…** and pick the file. Units come from the file. The editor shows
   the bodies and their faces, the model size and the suggested symmetry
   (from open exterior edges on X=0 / Y=0), plus warnings for covered air faces
   and exterior gaps.
2. Use the tree checkboxes to hide bodies or single faces; **Show all** undoes
   all hiding, **Clear** also clears the selection.
3. Set elements per wavelength (default 5), curvature and growth. The target
   element size is sound speed ÷ the project's maximum frequency ÷ elements per
   wavelength; the sound speed is that of the project's exterior air region
   (343 m/s when there is none yet).
4. Set the project symmetry to the suggested one, then **Generate** (F7);
   cancelling geometry generation stops it.

Each generation creates one tetrahedral mesh per touching group of air bodies
(one volume group per body), one exterior surface mesh, a region for each, a
rigid `walls` boundary for every ungrouped face, and an interface with matching
boundaries on both sides of every opening. Boundaries that already exist are
never resent, so roles, components and channels set in System survive
regeneration; mesh IDs stay stable. Under X or XY symmetry the model must
already be the half or quarter part; the host checks it lies in the positive
region. Saved projects reopen from their cached meshes without re-meshing.

Source units are metres; the STEP path is stored as given.
