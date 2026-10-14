# CAD Import

A bundled Generator Plugin that turns a STEP file into a conformal FEM/BEM mesh
assembly. Solids become **air bodies** (bounded regions); surface bodies become
**exterior surface bodies**, which together form the BEM exterior. Meshing runs
the same headless pipeline as `blab cad-import`, so both produce identical
meshes for identical inputs.

## Use

The plugin is enabled once when introduced. Select **CAD Import** as the default
in **Edit > Generator Plugins...** and add a design.

1. **Open STEP…** and pick the file. Units come from the file. The view draws
   the CAD faces smooth-shaded with their outlines (a display preview, not the
   generated mesh). The editor shows the bodies and their faces, the model
   size and the suggested symmetry
   (from open exterior edges on X=0 / Y=0), plus warnings for covered air faces
   and exterior gaps.
   A **closed shell** of surfaces (free faces that close up, as SolidWorks
   writes the solids of parts mixing solid and surface bodies) could be either,
   so the editor asks for each one after a quick scan, before the file is read
   fully once with the answers, with the shell highlighted and framed and
   the rest see-through: **Air volume** makes it an air body named
   after the shell, **Exterior surface** (also Escape) keeps it exterior. Change
   the choice later under **Closed shells**; another STEP file is asked afresh.
   The CLI takes `--closed-shell "Closed shell 1=air"` (unset shells are exterior).
2. Pick **Forward** and **Up**: the CAD axes that become Boundary Lab +Z and +Y.
   The view, size and suggested symmetry follow at once, without reading the
   file again; which faces open into air bodies is kept from the last read,
   and Generate uses the new axes.
3. Use the tree checkboxes to hide bodies or single faces; **Show all** undoes
   all hiding, **Clear** also clears the selection.
4. Click faces in the view (Ctrl/Shift+click adds or removes) or select them
   in the tree, enter a name and **Group selection** to make a **face group**
   (an existing name adds the faces; a face is in one group only). Faces named
   in the STEP file form groups of their name unless picked into another
   group. Groups store STEP names and fingerprints in CAD axes, so they
   survive changing Forward/Up. **Delete group** removes a picked group.
5. Set elements per wavelength (default 5), curvature and growth. The target
   element size is sound speed ÷ the project's maximum frequency ÷ elements per
   wavelength; the sound speed is that of the project's exterior air region
   (343 m/s when there is none yet). Override it in mm per body (double-click
   the tree's size column) or per face group (the list's size column).
6. Set the project symmetry to the suggested one, then **Generate** (F7);
   cancelling geometry generation stops it.

Each generation creates one tetrahedral mesh per touching group of air bodies
(one volume group per body), one exterior surface mesh, a region for each, a
rigid boundary for each new face group, a rigid `walls` boundary for every
ungrouped face, and an interface with matching
boundaries on both sides of every opening. Boundaries that already exist are
never resent, so roles, components and channels set in System survive
regeneration; mesh IDs stay stable. Boundaries of face groups that are gone
(a deleted group, a STEP name no longer present) are removed in the same
patch, with their interfaces; their components keep their other boundaries or,
left with none, are removed with their excitations.

**Re-import.** After the STEP file changes, a stored face is found again by its
STEP name when only one face carries it, else by its fingerprint: same surface
type and nearly the same centre, area and normal (tolerances in
`docs/specs/cad-import.md`, "Re-import matching"). A fingerprint matching
several faces matches none. A group missing any of its faces is marked **needs review** in the
face-group list and Generate is rejected, keeping the previous meshes. Pick
the faces again and **Group selection** under the group's name (this drops its
missing faces), or delete the group. When Boundary Lab regains focus and the
STEP file has changed on disk, the editor asks once whether to re-mesh.

Under X or XY symmetry the model must
already be the half or quarter part; the host checks it lies in the positive
region. Saved projects reopen from their cached meshes without re-meshing.

Source units are metres; the STEP path is stored as given.
