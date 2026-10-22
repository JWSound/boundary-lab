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
   and exterior gaps. While a file is read, the previous preview and controls
   stay; if reading fails they stay with the error, so the change can be undone.
   A **closed shell** of surfaces (free faces that close up, as SolidWorks
   writes the solids of parts mixing solid and surface bodies) could be either,
   so it is guessed: an air body, unless the file has no other exterior
   surface, in which case the largest closed shell is the exterior and the
   others air. Change it in the **Kind** dropdown on its body row (Air body /
   Exterior surface), which re-reads the file; another STEP file starts from
   the guesses again. The CLI takes `--closed-shell "Closed shell 1=air"`
   (unset shells get the guess).
2. Under **Placement**, rotate by any angle about the Boundary Lab X, Y and Z
   axes (in that order, about the origin), then translate in mm. The view, size,
   suggested symmetry and the detected interfaces follow at once, without
   reading the file again, as Generate will see them. **Show axes and planes**
   draws the labelled axes through the origin and the X=0 and Y=0 symmetry
   planes in green. The Meshes dialog keeps the generated meshes' scale and
   position locked: place them here.
   The CLI takes `--rotate X,Y,Z` (degrees) and `--translate X,Y,Z` (mm).
3. Use the tree checkboxes to hide bodies or single faces; **Show all** undoes
   all hiding, **Clear** also clears the selection.
4. Click faces in the view (Ctrl/Shift+click adds or removes) or select them
   in the tree; clicking a body row selects all its faces. The selection is
   orange, the face under the mouse pale orange (dark orange when selected).
   Enter a name and **New group** / **Add to "…"** to make a **face group**
   (an existing name adds the faces; a face is in one group only). Right-click
   a face for the same actions on the selection; a face outside the selection
   becomes the selection first. The list shows walls, then the interface
   groups, then the face groups. Faces named
   in the STEP file form groups of their name unless picked into another
   group. Groups store STEP names and fingerprints in CAD axes, so they
   survive changing the placement. **Delete group** removes a picked group.
5. Choose the **FEM mesh fineness**: FEM Fine (6 elements per wavelength at
   the maximum frequency, the default), FEM Medium (4), FEM Coarse (2) or
   Custom with a number; each level's tooltip names the frequency up to which
   it still gives 6 per wavelength. The **BEM mesh fineness** below it works
   the same for the exterior surface: BEM Fine (12), BEM Medium (5, the
   default for new designs), BEM Coarse (0.3) or Custom; a design saved
   without one uses its FEM value. FEM sizes air-body volumes and faces, BEM
   sizes exterior surface faces, and a face on both takes the finer. Each
   level's tooltip also gives its edge length at the project's maximum
   frequency. The target element size, one per method, is sound speed ÷
   the project's maximum frequency ÷ elements per wavelength; the sound
   speed is that of the project's exterior air region (343 m/s when there
   is none yet). A face or Volume row's Global names the level that sizes
   it (for example Global (BEM Medium)); a body, Faces or face group row
   names it when all its members share it, else it says plain Global.
   Curvature refinement (elements per full circle) and growth are set
   below it. **Size overrides** replace the target size as FEM Fine / Medium /
   Coarse, BEM Fine / Medium / Coarse or a custom size in mm: per body or face
   in the tree's **Mesh size** column (double-click), per face group in the
   list, or on the selected faces from
   the 3D right-click menu (**Mesh size ▸**). The most specific wins: face,
   face group, the finest of the face's bodies, then the mesh fineness; an
   empty cell's tooltip says where its size comes from. A face override whose
   face is gone after a re-import is dropped with a note. Designs saved before
   these levels keep their elements per wavelength (shown as Custom) and their
   mm overrides.
6. Set the project symmetry to the suggested one, then **Generate** (F7);
   cancelling geometry generation stops it. The status line shows how long
   the mesh job has been running; there is no automatic time limit, so cancel
   it yourself if it takes too long.

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
