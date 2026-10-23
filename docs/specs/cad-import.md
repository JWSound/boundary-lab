# CAD Import

## Problem Statement

Getting CAD geometry into Boundary Lab today means exporting each body to its own `.msh` file. On Autodesk Fusion that is done with the external Fusion2Msh add-in; other CAD tools such as SolidWorks have no equivalent path. After that, every physical group has to be wired up by hand in System. Meshes for FEM air volumes and the BEM exterior are produced separately, so their shared openings rarely match and need the `conform-interface` step. Mesh size has to be worked out by hand from the frequency range. Every CAD change repeats the whole process, and assignments made in System break when group names shift. Half and quarter models are common for loudspeakers, but generated multi-mesh assemblies cannot be used under symmetry at all.

## Solution

A bundled **CAD import** geometry design reads a STEP file and turns it into a complete mesh assembly.
- Solids are treated as **air bodies** (rear chambers, ports, waveguide throats) and become bounded regions.
- Surface bodies are treated as **exterior surface bodies** (cabinet and waveguide outsides) and together form the unbounded region.
- A **closed shell** of surfaces can be either, so the user chooses: an air volume or an exterior surface. SolidWorks writes the solids of parts that mix solid and surface bodies this way.
- All bodies are meshed in one conformal pass, so every interface between an air body and the exterior matches node for node. Interfaces are detected and created automatically.
- Mesh size follows the **target element size** rule: sound speed ÷ maximum frequency ÷ elements per wavelength, chosen as a **mesh fineness** per method: FEM (air bodies) and BEM (exterior surface), each a level (FEM Fine / Medium / Coarse, BEM Fine / Medium / Coarse) or Custom. **Size overrides** replace it on single faces and on air-body volumes; a size chosen for a body or **face group** is written to each of its faces.

The user works in a closable CAD Import dock:
- a 3D view on top, with face picking;
- a tree of bodies and faces with visibility controls;
- face-group naming;
- a **placement**: a free rotation and translation, shown against the labelled Boundary Lab axes (X horizontal, Y vertical, Z on axis) and the symmetry planes;
- sizing settings; the host's Generate (F7) meshes and its cancel stops meshing.

The resulting meshes appear in Meshes and the regions, boundaries and interfaces appear in System. The user only marks diaphragm face groups as moving and attaches components.

Re-importing an edited STEP file keeps every assignment whose faces can still be matched. Groups that can't be matched are listed for the user to resolve; nothing is reassigned silently. Reduced half and quarter models are supported under X and XY symmetry.

## User Stories

### Import

1. As a loudspeaker designer, I want to import a STEP file into a CAD Import design, so that I can use geometry from any CAD tool rather than only Fusion.
2. As a designer, I want every solid in the STEP file treated as an air body, so that I can model chambers, ports and throats directly as air.
3. As a designer, I want every surface body in the STEP file treated as an exterior surface body, so that cabinet and waveguide outsides form the BEM exterior.
4. As a designer, I want several exterior surface bodies (for example a cabinet and a separate waveguide surface) merged into one watertight exterior, so that the project keeps its single unbounded region.
5. As a designer, I want gaps or overlaps between exterior surface bodies reported with the edges that do not close, so that I know what to fix in CAD.
6. As a designer, I want STEP units honoured and the model's bounding-box size shown, so that a millimetre/metre mistake is obvious immediately.
7. As a designer, I want body names from the STEP file shown in the tree, so that I recognise my CAD bodies.
8. As a designer, I want unnamed bodies to get stable, readable default names, so that STEP exports without names are still usable.

### Closed shells

61. As a designer, I want each closed shell of surfaces to get a guessed kind (air body or exterior surface) that I can change in a Kind dropdown on its body row, without being asked in a dialog, so that SolidWorks exports (which write solids as closed shells) and closed waveguide outsides both import correctly without interruption.
62. As a designer, I want a closed shell guessed as an air body, unless the file has no other exterior surface body, in which case the largest closed shell is the exterior, so that the usual exports need no correction.

### Orientation

9. As a designer, I want the Boundary Lab axes labelled X · Horizontal, Y · Vertical and Z · On Axis in the 3D view, so that I can rotate my model's front onto the acoustic reference axis and its up onto vertical.
11. As a designer, I want the 3D view, symmetry detection and interface detection to work on the reoriented geometry, so that what I see is what gets meshed.
12. As a designer, I want the CAD origin kept as the Boundary Lab origin unless I translate the geometry, so that I can control the acoustic reference point in CAD.
63. As a designer, I want to rotate the geometry by any angle about the Boundary Lab X, Y and Z axes and translate it in millimetres before meshing, so that geometry not modelled at the acoustic reference position or axis can still be placed correctly.
64. As a designer, I want the Boundary Lab axes, always longer than the model, and the symmetry planes X=0 and Y=0 shown in the 3D view, with a toggle to hide them, so that I can see where the geometry sits relative to the origin and the symmetry planes.
86. As a designer, I want both symmetry planes in one green, distinct from the Y axis, so that the planes read as one kind of thing.
87. As a designer, I want the axis labels far enough outside the model that they do not sit over it, so that the model stays readable.
88. As a designer, I want a placement change to update the detected interfaces and their face groups in the preview, so that the preview always shows what Generate will create.
65. As a designer, I want placement changes to move the displayed geometry immediately without re-reading the STEP file, so that adjusting the placement is quick.
66. As a designer, I want scale and position of generated meshes fixed in Meshes, so that the geometry is placed in exactly one place, its design.

### Bodies, meshes and interfaces

13. As a designer, I want air bodies that touch each other meshed together into one tetrahedral mesh with one volume group per body, so that a chamber and its port form one continuous air domain.
14. As a designer, I want an isolated air body to get its own mesh, so that separate chambers stay separate regions.
15. As a designer, I want all bodies meshed in one conformal pass, so that FEM and BEM interface faces match node for node without `conform-interface`.
16. As a designer, I want a face where an air body fills an opening in the exterior detected as an interface, so that I don't have to tag port mouths and throats by hand.
17. As a designer, I want a matching face group created on both sides of each detected interface, so that both regions have an interface boundary.
18. As a designer, I want the interface itself created in System, so that the coupled model is ready without opening the Interfaces tab.
19. As a designer, I want a warning when an exterior surface covers an air-body face continuously instead of having an opening, so that I notice a forgotten cut in CAD.
20. As a designer, I want one region created per connected air-body group and one for the exterior, so that System is pre-filled.
21. As a designer, I want every face not in any group to fall into a default walls group with a rigid boundary, so that nothing is left unassigned.

### Face groups and sources

22. As a designer, I want to pick faces in the 3D view and name them as a face group, so that diaphragms and other special surfaces get their own physical group.
23. As a designer, I want a face group to span several faces (for example cone and dust cap), so that a driver's whole radiating surface is one group.
24. As a designer, I want face names from the STEP file used as face groups when present, so that groups named in CAD carry over.
25. As a designer, I want new face groups to arrive in System as rigid boundaries, so that I decide which ones move.
26. As a designer, I want to mark a diaphragm face group as moving and attach an ideal velocity source or electrodynamic transducer in System, so that it becomes a source.
27. As a designer, I want boundary roles, relative motion weights, components and channel routing to stay mine, so that regenerating never overwrites them.
69. As a designer, I want to click a face in the 3D view to select it, so that I can pick faces where I see them.
70. As a designer, I want to choose a current face group and add single selected faces to it or remove them from their group, so that I can build and fix groups face by face.
71. As a designer, I want to right-click a face in the 3D view to move it to another face group or to walls, with the group it is already in left out, so that grouping one face is one step and the menu never offers a move that does nothing.
74. As a designer, I want a right-clicked face marked in the view with its group named and its row selected, so that I can check which group a face is in.
75. As a designer, I want to rename an interface face group from the right-click menu of one of its faces, so that its boundaries in System carry a name I recognise.
76. As a designer, I want faces I grouped before they became interface faces moved to their interface face group with a note, so that generation does not fail on them.
72. As a designer, I want each face group shown in its own colour in the 3D view, so that I see at a glance what is grouped.
73. As a designer, I want the walls face group and the interface face groups listed first, before the picked and STEP-named face groups, so that I see every surface group the mesh will have, the automatic ones first, and can size them.
77. As a designer, I want selected faces, including a right-clicked face, drawn in orange, so that there is one selection colour.
78. As a designer, I want a face under the mouse drawn pale orange, or dark orange when it is selected, so that I see which face a click will act on.
79. As a designer, I want right-clicking a selected face to keep the whole selection orange and act on all of it, so that I can group several Ctrl+clicked faces in one step.
80. As a designer, I want right-clicking a face outside the selection to make it the selection, so that the menu never acts on faces I am not pointing at.
81. As a designer, I want clicking a body, Faces, Volume or face-group row to select all its faces (a volume's faces are those around it), so that I see the whole body or group in orange and can act on it.
90. As a designer, I want right-clicking a face row, a body's Faces row or a face-group row to open the same menu as right-clicking in the 3D view, its faces becoming the selection first unless the face is already in it, so that I can act where I found the faces.
91. As a designer, I want the 3D view to open looking along −Z (Z towards me, Y up, X to the right), so that the model appears as the balloon and mesh views show it.

### Visibility

28. As a designer, I want to show or hide each body in the tree, so that I can see inside nested geometry.
29. As a designer, I want to show or hide individual faces, so that I can reach faces hidden behind others.
30. As a designer, I want a Show all action, so that I can quickly undo all hiding.
31. As a designer, I want a Clear action that shows everything and clears the current selection without deleting face groups, so that I can start a new selection safely.
32. As a designer, I want to close the CAD Import dock when I'm not using it, so that it doesn't take up space.

### Mesh sizing

33. As a designer, I want the default target element size computed as sound speed ÷ maximum frequency ÷ elements per wavelength, so that the mesh suits my frequency range.
34. As a designer, I want the project's sound speed and maximum frequency used, so that sizing follows the rest of the project.
35. As a designer, I want to choose the FEM mesh fineness as FEM Fine (6 elements per wavelength at fmax, the default), FEM Medium (4), FEM Coarse (2) or a custom number, and the BEM mesh fineness as BEM Fine (12), BEM Medium (5, the default), BEM Coarse (0.3) or a custom number, each level saying up to which frequency it holds 6 per wavelength, so that I can trade accuracy for speed knowingly.
36. As a designer, I want the computed FEM and BEM target element sizes shown next to the inputs, so that I know the resulting edge lengths.
37. As a designer, I want the FEM target element size applied to air bodies (volumes and faces) and the BEM one to the exterior, with the finer one on interface faces, so that the interior is meshed uniformly, the exterior can be graded, and both sides of each interface match.
38. As a designer, I want to set the size of an air body's volume (its interior) separately from its faces, so that large uncritical volumes can be coarser.
39. As a designer, I want choosing a size for a body or a face group to set it on each of its faces (and a body's volume), so that diaphragms and throats can be finer in one step.
82. As a designer, I want to set the size for any selected faces without making them a face group, so that the back of a surface can be coarser than the front of the same group.
83. As a designer, I want sizes chosen as Global, FEM Fine / Medium / Coarse, BEM Fine / Medium / Coarse or a custom size in mm, on any face or volume, from a Mesh size dropdown always shown on every tree and face-group row and from the 3D right-click menu, so that I size where I see the geometry.
84. As a designer, I want only faces and volumes to carry a size, every face and volume row to show its own size or Global (with the level of the mesh fineness that sizes it), and every body, Faces and face-group row to show the size its members share or Mixed, so that what a row shows is what gets meshed.
85. As a designer, I want existing designs to keep their elements per wavelength (for both FEM and BEM) and mm overrides, their body and face-group sizes turned into the face and volume sizes that applied before, so that their meshes do not change.
92. As a designer, I want each body row to hold a Volume row (air bodies only) and a Faces row with the faces under it, so that I see where the volume size is set.
93. As a designer, I want the Kind column to name every body's kind, with a dropdown only on closed shells, so that every body's role is visible.
40. As a designer, I want curvature-based refinement, always on, with its elements per full circle editable, so that curved surfaces such as waveguide profiles are captured accurately while flat areas keep the wavelength size.
41. As a designer, I want element size to change gradually, with a limited growth rate between fine and coarse areas, so that the mesh has no abrupt size jumps.
42. As a designer, I want to start meshing with one button and cancel it, so that I stay in control of long runs.

### Symmetry

43. As a designer, I want to model a half or quarter part in CAD, so that solves run faster.
44. As a designer, I want open edges lying on the X=0 or Y=0 planes detected, so that CAD Import can suggest the matching project symmetry.
45. As a designer, I want generation rejected with a clear message when the project symmetry does not match the geometry, so that I don't get a wrong solve.
46. As a designer, I want generated assemblies that are already reduced accepted under X and XY symmetry, so that coupled half and quarter models can be solved.
47. As a provider author, I want to declare that my assembly is already reduced and have the host check that it lies in the positive region, so that any assembly provider can support symmetry.

### Re-import

48. As a designer, I want to re-mesh after changing the STEP file and keep assignments whose faces still match, so that iterating on CAD is cheap.
49. As a designer, I want faces matched by STEP name first, so that named faces survive any geometric change.
50. As a designer, I want unnamed faces matched by a geometric fingerprint (centre, area, normal, surface type) within a tolerance, so that small CAD edits don't lose assignments.
51. As a designer, I want face groups that cannot be matched listed as needing review, with the previous meshes kept, so that nothing is reassigned silently.
52. As a designer, I want to re-pick faces for a group that needs review, so that I can repair it.
53. As a designer, I want to delete a group that needs review together with its boundary, detaching it from its component, so that removal is one consistent step.
54. As a designer, I want CAD Import to notice when the STEP file changes on disk and offer to re-mesh, so that I don't work with stale meshes.
55. As a designer, I want mesh IDs to stay stable across regeneration, so that references in System, Meshes and Channels survive.

### Project and automation

56. As a designer, I want the CAD Import design (STEP path, orientation, face groups, sizing) saved in the project, so that reopening restores my setup.
57. As a designer, I want generated meshes restored when reopening a project without re-meshing, so that opening is fast.
58. As a designer, I want the generated meshes listed in Meshes, locked to their design, so that I can see what was produced.
67. As a designer, I want opening a STEP file to read the geometry fully only once, so that opening is fast.
89. As a designer, I want the previous preview and its controls to stay while the STEP file is read, and to stay with the error shown if reading fails, so that I can always undo the change that caused it.
68. As a designer, I want the 3D view to show smooth CAD faces with their outlines rather than a triangle mesh, so that the preview is not mistaken for the generated mesh.
59. As a power user, I want a CLI command that meshes a STEP file with a given maximum frequency and elements per wavelength, so that I can script or batch mesh generation.
60. As a power user, I want the CLI and the dock to produce identical meshes for the same inputs, so that scripted and interactive work agree.

## Implementation Decisions

- **CAD import is a bundled geometry provider** that returns a named mesh assembly together with the System entries in one response, packaged like the bundled vented-enclosure provider. It uses the host's existing assembly contract: `fem_volume` and `bem_surface` resources, stable host mesh IDs, transactional acceptance and atomic rejection. The editor is a custom provider editor that the host shows inside its design dock.
- **Host change (slice 0):** generated assemblies are currently rejected unless project symmetry is Off.
  - An assembly may now declare itself already reduced. The host then checks every resource against the positive X, or positive X/Y, region using the existing symmetry checks, instead of rejecting it.
  - The host never cuts assembly meshes. Assemblies that do not declare themselves reduced keep today's behaviour.
  - This is part of the provider API and must be documented in the provider API guide.
- **Design source**, versioned through the provider's own schema version, holds:
  - the STEP file path;
  - the placement: a rotation in degrees about the Boundary Lab X, Y and Z axes, and a translation in millimetres;
  - the interface names: detected interface name → the user's name;
  - elements per wavelength, for FEM (`elements_per_wavelength`) and BEM (`bem_elements_per_wavelength`; when missing, the FEM value), the two mesh finenesses; levels are not stored separately;
  - size overrides per face (`face_sizes`, each with its face reference, like a face group's) and per air-body volume (`volume_sizes`, keyed by body name), each either elements per wavelength or an edge length; body and face-group sizes (`body_sizes`, `face_group_sizes`) are only read, to convert older designs;
  - curvature and growth settings;
  - the face groups: each a name, its member faces' STEP names if present, and geometric fingerprints (taken in CAD axes, before the placement, so changing the placement keeps the groups);
  - the closed shell choices: closed shell name → `air` or `exterior`.
- **Meshing pipeline** (headless, in the provider's worker process):
  1. Read the STEP file with Gmsh's OpenCASCADE kernel, in millimetres (OpenCASCADE's fixed tolerance is too coarse in metres); results are reported and written in metres.
  2. Apply the placement, as rigid motions so exact surface types are kept: the rotation about the fixed Boundary Lab axes in the order X, Y, Z about the origin, then the translation.
  3. Classify solids as air bodies and surface bodies as exterior surface bodies. Before that, free faces that close up among themselves through shared edges (each edge used by exactly two of them, as imported, before fragmenting) form a closed shell. Closed shells are named from the STEP if possible, else "Closed shell N" in import order. A closed shell chosen as air becomes a solid (an air body named after the shell); one chosen as exterior stays exterior surface. One without a choice gets the guessed kind: air, unless the file has no exterior surface body other than closed shells, in which case the closed shell with the largest bounding box is exterior and the others air. A choice for a name the file does not contain is rejected.
  4. Fragment all bodies into shared topology.
  5. Detect interfaces and symmetry-plane open edges.
  6. Mesh once, with size fields from the target element size, the face and volume overrides, curvature and the growth limit.
  7. Split the result into one tetrahedral mesh per connected air-body group (one volume group per air body) and one BEM surface mesh for the exterior.
- **Sizing rule:** a target element size is c ÷ fmax ÷ N, with c and fmax taken from the request's project configuration; there is one per method (ADR 0002). The FEM mesh fineness sets N for air-body volumes and the faces of air bodies: FEM Fine 6 (the default for new designs), FEM Medium 4, FEM Coarse 2 or a custom number. The BEM mesh fineness sets N for the faces of exterior surface bodies: BEM Fine 12, BEM Medium 5 (the default for new designs), BEM Coarse 0.3 or a custom number; the BEM levels are 1.5, 4 and 60 mm edge lengths at a 20 kHz reference, rounded so no two levels share an N. A face on both (an interface) gets the finer target size. A stored N equal to a level's shows as that level, any other as Custom, so existing designs (N = 5, no BEM fineness, which then takes the FEM N) keep their meshes. Each level's tooltip names the frequency up to which it still gives 6 elements per wavelength (fmax × N ÷ 6). Curvature refinement is always on, defaulting to about 12 elements per full circle (editable); it only refines, flat areas keep the target size, and there is no coarser-than-wavelength "dynamic" mode. Growth is at most 1.3× per element, and the smallest override sets the global floor. Small-feature refinement is out of scope.
- **Size overrides:** a size override is a level (its N, giving c ÷ fmax ÷ N) or an edge length in mm. The editor stores them only on faces and on air-body volumes; nothing is inherited. Choosing a size on a row writes it to its members: a body row to its volume and every face, a Faces row or a face-group row (walls and interfaces included) to its faces, a Volume row to the volume; Global clears them. Other rows sharing those faces may then show Mixed. A face moved into a group, or new after a re-import, keeps its own size or none. A face or volume without one gets its target element size (FEM, BEM, or the finer for an interface face). Any level, FEM or BEM, can be chosen on any face or volume. The pipeline, the provider backend and the editor size through one Qt-free sizing module (target element size, override shape and text, shared size or Mixed, writes to members, dropped sizes, which size wins). A stored face override that no longer matches a face after a re-import is dropped with a note in the editor; it never blocks generation. Existing mm overrides are kept as edge lengths.
  - The headless pipeline takes volume sizes, which size only the interior of an air body, never its faces. It keeps accepting body, face-group and face overrides (most specific wins on a face: face, face group, finest body; a body override also sizes its volume) for designs the editor has not converted yet; the editor writes only face and volume sizes. A volume size for a name that is not an air body is rejected. The CLI is unchanged.
  - Migration: a design holding body or face-group sizes is converted when the editor loads it (once its geometry and the maximum frequency are known; never as a side effect of drawing), by a pure function of the sizing module, to the same mesh: each face gets the size that won before (face, face group, finest body) unless that is none; an air body's size becomes its volume size; body and face-group sizes are then dropped.
- **Interface detection:** a fragmented face shared by an air body and an exterior surface body is an interface. A matching face group is created on both sides, both boundaries get the interface role, and the interface entity is created. Connected interface faces of one air body form one interface named `Interface <n> <air body>` (n counting from 1 per air body); the user can rename it, and the name is kept while that interface is detected. A name must not be `walls`, another interface's or a face group's. An air-body face lying under a continuous exterior surface produces a warning, not an interface. Touching air bodies are merged into one mesh rather than creating interfaces between FEM regions, which the solver does not support.
- **Face groups:** a stored face resolves to one face: the only face carrying its STEP name; if several faces carry that name, the one among them whose fingerprint matches; if none does, the face whose fingerprint matches. A face can be in one face group only; faces of a detected interface always stay in its interface face group: a picked group holding one (picked before the interface was detected) keeps it stored, but while the interface is detected the face is only in the interface face group, with a note in the editor; a picked group left with no other face is listed with no faces (so it can be deleted) and has no surface group. The editor and the pipeline resolve face groups through one Qt-free module, so they always agree. Faces with a STEP name that are in no picked group form a face group of that name; `NONE`, SolidWorks' placeholder, counts as no name. Group names `walls` and interface names are reserved.
- **What the provider sends on each generation:**
  - It always sends the meshes, regions, face groups and detected interfaces.
  - It sends boundary entries only for face groups that are new, defaulting to rigid; ungrouped faces form a default walls group.
  - It never resends existing boundaries.
  - Roles, weights, components, excitations and channels belong to the user and survive regeneration through the host rule that omitted fields are preserved.
- **Re-import matching:** faces are matched by a unique STEP name, then by fingerprint within a tolerance (among the faces sharing the STEP name, if several do): same surface type, centre within 1 % of the bounding-box diagonal, area within 5 %, normal pointing the same way within about 2.6° (a flipped normal does not match). A stored face whose fingerprint matches more than one face is ambiguous and counts as not found. If any face group is missing any of its stored faces (none or only some found, so nothing is reassigned silently), generation is rejected before the response is sent with one message naming every such group; the previous meshes stay in place. The editor runs the same matching on the loaded file and lists those groups as needing review. Grouping a selection under the name of a group that needs review drops its missing faces, which repairs it. Deleting a group, like any face group that disappears, sends on the next generation the explicit removal of its boundary, of interfaces using it, and its detachment from its component in the same patch; a component left without boundaries is removed with its excitations. A component that also holds boundaries of another design is left as is, so the host rejects the generation until the user detaches the boundary in System. Bounded regions whose mesh is gone are removed too.
- **Watching the STEP file:** when the application regains focus, a changed STEP file (modification time or size) triggers a re-mesh prompt, once per change, rather than an automatic regeneration. The file is stamped before it is read, so an edit while it loads still prompts; a design without a STEP file never prompts. Yes reloads the preview and starts the host's Generate.
- **Symmetry suggestion:** open edges lying entirely on X=0 or Y=0 after the placement suggest the matching project symmetry. A mismatch with the project setting rejects generation with a message.
- **Editor layout**, top to bottom:
  1. the STEP file row, and an info line holding only the Size / Symmetry suggestion (in the normal text colour; reading and read errors show there too);
  2. the 3D view, on top of a draggable splitter;
  3. below the splitter, in a plain panel with no outer scroll box (each list scrolls on its own): the **warnings** (each `Warning: …` line and the "N face(s) cannot be meshed; see the list." line, in red, a darker red on a light theme and a lighter one on a dark theme, hidden when there are none), the Unmeshable faces list (hidden when there are none), the Placement box, and the Bodies and faces box (the body and face tree with Show all / Clear / Show axes and planes);
  4. the Face groups box, the Mesh sizes box (mesh fineness and sizing settings), and the status line (normal colour; only the warnings are red).

  Each group of related settings or list sits in its own framed, titled box: Unmeshable faces, Placement, Bodies and faces, Face groups and Mesh sizes.
- **Editor:** a 3D viewport with face picking above a body and face tree, with visibility checkboxes, Show all, and Clear (which shows everything and clears the selection, never deleting groups). A click in the view selects a face, Ctrl or Shift+click adds or removes one (clicks are detected from the widget's mouse events, since the camera interaction holds VTK's mouse focus between press and release; a press and release without movement is a click, a drag rotates); the selection is shared with the tree and highlighted in the view, with a hint line and a selected-face count. Each face group is drawn in its own colour (a matching swatch in the list), ungrouped faces in the plain face colour, the selection on top in orange. The face under the mouse is drawn pale orange, or dark orange when selected (view only; tree rows do not hover). There is no other highlight colour, apart from the see-through grey context faces around a shown unmeshable face (below) and the see-through red fill of unmeshable faces (see Preview). The view opens, and resets on loading a file, looking along −Z: Z towards the viewer, Y up, X to the right; placement changes keep the camera.
  - Tree: each body row holds a **Volume** row (air bodies only; no checkbox, not drawn) and a **Faces** row with the body's face rows under it. Clicking a body, Volume or Faces row selects the body's faces (for a volume, the faces around it). Right-clicking a face, Faces or face-group row opens the face menu on its faces (they become the selection unless the right-clicked face is in it); right-clicking a Volume row offers only Mesh size.
  - **Mesh size** column: a dropdown always shown on every body, Volume, Faces and face row and every face-group row: **Global (<level>)** / FEM Fine / FEM Medium / FEM Coarse / BEM Fine / BEM Medium / BEM Coarse / Custom… mm. A face or Volume row shows its own size or Global with the level of the mesh fineness that sizes it (FEM, BEM, or the finer on an interface face); a body, Faces or face-group row's Global names that level when all its members share it, else plain Global; a body, Faces or face-group row shows the size its members share, or **Mixed**. Choosing writes the size to the members (see Size overrides).
  - **Kind** column: every body row names its kind; closed-shell body rows have a dropdown (Air body / Exterior surface), whose change re-reads the file, other bodies plain text.
  - **Unmeshable face in context:** clicking a row of the Unmeshable faces list hides nothing and leaves the tree's checkboxes alone. The bodies containing the face are drawn as usual; every other visible face is a **context face**, drawn see-through grey and still pickable. The face is selected, named with its group in the status line, its group's row selected, its curves outlined in orange (with or without a fill, see Preview), and the camera frames them (its fill when it has no curves; with neither, the camera stays). Clicking another row moves the context; picking a context face selects it and keeps the context; Show all and Clear end it, Clear also emptying the selection. Unchecked faces stay hidden, never grey.

  Below that sit the face-group list and its controls:
  - Rows: a fixed **walls** row first (faces in no other group; selecting it selects them; not deletable or renamable), then the interface groups (selectable, sizable, renamed only from the face menu, never deleted), then the face groups and groups named in the STEP file. Every row has a face count and a Mesh size dropdown.
  - Clicking a row selects its faces and makes it the **current group**, filling the name field. The button reads **Add to "<name>"** for an existing name, **New group** for a new one. Adding to walls is the same as Remove from group.
  - **Remove from group** moves the selected faces back to walls. **Delete group** removes a picked group.
  - Right-click without movement on a face in the view: on a face outside the selection, that face becomes the selection; then the selection is drawn orange (nothing see-through), the face and its group are named in the status line (`Face 64 · Group '…'`) and the group's row is selected without changing the face selection, and a flat menu opens: **Move to "<name>"** for each picked or STEP-named group except one already holding every face acted on, *New group…* (asks the name), **Move to walls** unless they are all in walls, and **Mesh size ▸** Global (<level>) / the six levels / Custom… (a face size override; Global clears it). It acts on the whole selection; interface faces in it are skipped with a note, except for Mesh size. On an interface face the menu offers *Rename interface…* and Mesh size. Right-drag still pans.

  The Placement box (Rotate X/Y/Z in degrees, Translate X/Y/Z in millimetres, number fields only) sits above the tree, close to the view; further down sit the sizing settings: an FEM mesh fineness dropdown (FEM Fine / Medium / Coarse / Custom with a number field), a BEM mesh fineness dropdown (BEM Fine / Medium / Coarse / Custom with a number field), curvature elements per circle, and the computed FEM and BEM target element sizes. Meshing uses the host's Generate (F7) and geometry-generation cancel. Closed shells are never asked about: they get the guessed kind, stored as a choice only when the user changes the Kind dropdown.
- **Preview:** a display tessellation of the CAD faces, drawn smooth-shaded with the face outlines (edges between faces), never the triangle edges; it is not the generated mesh. A face Gmsh cannot mesh gets **fill** triangles instead where its area reads positive: points of a grid over its parameters that lie inside the trimmed face, two triangles per grid cell with all four corners inside (a coarse grid first finds where the face lies within its parametrisation bounds, a finer one fills that box). Other Gmsh 2D algorithms fail on such faces the same way, so none are tried. A face whose area reads zero or negative keeps no triangles, since the inside test is unreliable there. The fill goes into the preview's triangles like any other face's; the face stays unmeshable and Generate still rejects it. Unmeshable faces with a fill are always drawn see-through red, in the normal view and in context, even when hovered or selected, and outlined in orange when selected; a face without one is only outlined. The view shows the Boundary Lab axes through the origin, labelled X · Horizontal, Y · Vertical and Z · On Axis, and the see-through symmetry planes X=0 and Y=0, both in one green distinct from the Y axis; planes reach past the model and the origin by a fifth of its span along each axis (at least a tenth of its largest size), axes by at least half its span so their labels (drawn on top) sit clear of the model, all hidden by one toggle. Opening a file reads it fully once with the guessed closed-shell kinds. Placement changes are rigid motions and only transform the displayed preview, without re-reading: its size, suggested symmetry, warnings and the detected interfaces (with their interface face groups) follow, the opening test being redone on the moved edges, so the preview always matches Generate. Closed shell choices and a changed or new file re-read the STEP file. While a file is read the previous preview, tree and controls stay, marked as reading; if the read fails they stay with the error shown, so the change that caused it can be undone. The placement is applied to the geometry in the meshing run only.
- **Meshes:** scale and translation of generated assembly rows (any provider) are locked at the design's values; a generated geometry is placed only by its design. Choosing another STEP file clears the stored choices. The editor follows host revisions and busy state like the vented-enclosure editor.
- **CLI:** a thin command over the same headless pipeline taking a STEP path, the maximum frequency, elements per wavelength (FEM, and `--bem-elements-per-wavelength`, defaulting to the FEM value) and an output folder, and writing the assembly meshes. A repeatable `--closed-shell NAME=air|exterior` sets closed shell choices.
- **Licence:** Fusion2Msh (CC BY-NC 4.0) informed the design only. Its code must not be copied into Boundary Lab, which is GPL-3.

## Testing Decisions

- **Good tests check external behaviour only:** the meshes produced and their physical groups, edge lengths, conformity at interfaces, the System entries that compile, rejections and their messages, and user edits surviving regeneration. No tests on internal helpers.
- **Primary seam: the provider through the public generator contract.**
  - Load the plugin through the provider catalog, generate from a STEP fixture and design source, stage the result, then compile the physical system or prepare a solve.
  - Prior art: the vented-enclosure generator tests.
  - Covers sizing (through the edge lengths of generated meshes), interface detection, placement, symmetry detection, System entries, and re-import matching (regenerating from a modified fixture).
- **Slice 0** uses the same contract with the existing test assembly fixture: an already-reduced variant accepted under X and XY symmetry, a resource outside the positive region rejected atomically, and unreduced assemblies still rejected. Prior art: the generator assembly tests.
- **UI seam:** the main window with the CAD Import dock. Generate from the dock, check the visibility tree and Show all / Clear, and drive face picking through the editor's API rather than mouse clicks. Prior art: the vented-enclosure UI tests.
- **CLI:** a thin end-to-end run on a fixture. Prior art: the CLI tests.
- **Fixtures:**
  - SolidWorks AP214 exports in millimetres of a vented box (chamber, port, exterior) and a waveguide (throat volume, exterior), committed if small.
  - A test helper strips names to exercise matching by fingerprint.
  - Until the files exist, geometry is built in code with Gmsh's OpenCASCADE kernel and written to STEP.

## Out of Scope

- Small-feature or edge-proximity refinement (deferred to v2).
- Dragging the geometry in the 3D view to place it (placement is set by number fields).
- Host cutting of full assemblies under symmetry; only assemblies that are already reduced are accepted.
- Interfaces between two FEM regions; touching air bodies are merged instead.
- Treating solids as cabinet material and deriving the air by subtraction.
- Assigning boundary roles or components inside CAD Import; that stays in System.
- Changing the generator-provider contract beyond the symmetry declaration.

## Further Notes

- **Delivery:** the slices below on one feature branch, opened as separate PRs once the whole feature is built:
  0. Host symmetry change.
  1. Headless core and CLI.
  2. Provider package, 3D editor and visibility tree.
  3. Face picking, face groups, orientation pickers and System entries.
  4. Re-import matching and watching the STEP file.
  4a. Click picking fix and face-group workflow.
  5. Preview look and closed shell highlight.
  6. Opening speed.
  7. Placement, axes and origin planes, locked Meshes placement.
  8. Editor polish: selection and hover colours, right-click on a selection, body selection, list order, green planes, axis labels, Kind dropdown instead of the closed-shell dialog, a never-blank reading preview, interfaces re-detected on placement.
  9. Mesh fineness and size overrides on bodies, face groups and faces.
  10. Sizes on faces and volumes only (Volume and Faces rows, Global / Mixed, always-shown dropdowns, migration), Move menu, row right-click, Kind text, default −Z view.
  11. Separate FEM and BEM levels and mesh finenesses (ADR 0002).
- **Vocabulary** follows the root glossary: air body, exterior surface body, face group, target element size, CAD import. A "source" is a moving boundary owned by a component.
