# CAD Import

## Problem Statement

Getting CAD geometry into Boundary Lab today means exporting each body to its own `.msh` file. On Autodesk Fusion that is done with the external Fusion2Msh add-in; other CAD tools such as SolidWorks have no equivalent path. After that, every physical group has to be wired up by hand in System. Meshes for FEM air volumes and the BEM exterior are produced separately, so their shared openings rarely match and need the `conform-interface` step. Mesh size has to be worked out by hand from the frequency range. Every CAD change repeats the whole process, and assignments made in System break when group names shift. Half and quarter models are common for loudspeakers, but generated multi-mesh assemblies cannot be used under symmetry at all.

## Solution

A bundled **CAD import** geometry design reads a STEP file and turns it into a complete mesh assembly.
- Solids are treated as **air bodies** (rear chambers, ports, waveguide throats) and become bounded regions.
- Surface bodies are treated as **exterior surface bodies** (cabinet and waveguide outsides) and together form the unbounded region.
- A **closed shell** of surfaces can be either, so the user chooses: an air volume or an exterior surface. SolidWorks writes the solids of parts that mix solid and surface bodies this way.
- All bodies are meshed in one conformal pass, so every interface between an air body and the exterior matches node for node. Interfaces are detected and created automatically.
- Mesh size follows the **target element size** rule: sound speed ÷ maximum frequency ÷ elements per wavelength. It can be overridden per body and per **face group**.

The user works in a closable CAD Import dock:
- a 3D view on top, with face picking;
- a tree of bodies and faces with visibility controls;
- face-group naming;
- orientation pickers that map the CAD axes onto Boundary Lab's +Z forward and +Y up;
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

61. As a designer, I want to be asked whether each closed shell of surfaces is an air volume or an exterior surface, and to change my answer later, so that SolidWorks exports (which write solids as closed shells) and closed waveguide outsides both import correctly.

### Orientation

9. As a designer, I want to choose which CAD axis points forward (mapped to Boundary Lab +Z), so that my model's front faces the acoustic reference axis.
10. As a designer, I want to choose which CAD axis points up (mapped to Boundary Lab +Y), so that horizontal and vertical polars mean what I expect.
11. As a designer, I want the 3D view, symmetry detection and interface detection to work on the reoriented geometry, so that what I see is what gets meshed.
12. As a designer, I want the CAD origin kept as the Boundary Lab origin, so that I control the acoustic reference point in CAD.

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

### Visibility

28. As a designer, I want to show or hide each body in the tree, so that I can see inside nested geometry.
29. As a designer, I want to show or hide individual faces, so that I can reach faces hidden behind others.
30. As a designer, I want a Show all action, so that I can quickly undo all hiding.
31. As a designer, I want a Clear action that shows everything and clears the current selection without deleting face groups, so that I can start a new selection safely.
32. As a designer, I want to close the CAD Import dock when I'm not using it, so that it doesn't take up space.

### Mesh sizing

33. As a designer, I want the default target element size computed as sound speed ÷ maximum frequency ÷ elements per wavelength, so that the mesh suits my frequency range.
34. As a designer, I want the project's sound speed and maximum frequency used, so that sizing follows the rest of the project.
35. As a designer, I want elements per wavelength to default to 5 and be editable, so that I can trade accuracy for speed.
36. As a designer, I want the computed target element size shown next to the inputs, so that I know the resulting edge length.
37. As a designer, I want the same target element size applied to the air bodies and the exterior, so that both sides of each interface are sized consistently.
38. As a designer, I want to override the size for one air body or exterior surface body, so that large uncritical volumes can be coarser.
39. As a designer, I want to override the size for one face group, so that diaphragms and throats can be finer.
40. As a designer, I want curvature-based refinement, so that curved surfaces such as waveguide profiles are captured accurately.
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
  - the axis mapping (Forward and Up, each one of ±X/±Y/±Z);
  - elements per wavelength;
  - size overrides per body and per face group;
  - curvature and growth settings;
  - the face groups: each a name, its member faces' STEP names if present, and geometric fingerprints (taken in CAD axes, before the axis mapping, so changing Forward or Up keeps the groups);
  - the closed shell choices: closed shell name → `air` or `exterior`.
- **Meshing pipeline** (headless, in the provider's worker process):
  1. Read the STEP file with Gmsh's OpenCASCADE kernel, in metres.
  2. Apply the axis mapping.
  3. Classify solids as air bodies and surface bodies as exterior surface bodies. Before that, free faces that close up among themselves through shared edges (each edge used by exactly two of them, as imported, before fragmenting) form a closed shell. Closed shells are named from the STEP if possible, else "Closed shell N" in import order. A closed shell chosen as air becomes a solid (an air body named after the shell); one without a choice, or chosen as exterior, stays exterior surface. A choice for a name the file does not contain is rejected.
  4. Fragment all bodies into shared topology.
  5. Detect interfaces and symmetry-plane open edges.
  6. Mesh once, with size fields from the target element size, the overrides, curvature and the growth limit.
  7. Split the result into one tetrahedral mesh per connected air-body group (one volume group per air body) and one BEM surface mesh for the exterior.
- **Sizing rule:** the target element size is c ÷ fmax ÷ N, with c and fmax taken from the request's project configuration and N defaulting to 5. The same size applies to FEM and BEM. Curvature defaults to about 12 elements per full circle, growth to at most 1.3× per element, and the smallest override sets the global floor. Small-feature refinement is out of scope.
- **Interface detection:** a fragmented face shared by an air body and an exterior surface body is an interface. A matching face group is created on both sides, both boundaries get the interface role, and the interface entity is created. An air-body face lying under a continuous exterior surface produces a warning, not an interface. Touching air bodies are merged into one mesh rather than creating interfaces between FEM regions, which the solver does not support.
- **Face groups:** a stored face resolves to one face: the only face carrying its STEP name; if several faces carry that name, the one among them whose fingerprint matches; if none does, the face whose fingerprint matches. A face can be in one face group only; faces of a detected interface can't be grouped. Faces with a STEP name that are in no picked group form a face group of that name; `NONE`, SolidWorks' placeholder, counts as no name. Group names `walls` and interface names are reserved.
- **What the provider sends on each generation:**
  - It always sends the meshes, regions, face groups and detected interfaces.
  - It sends boundary entries only for face groups that are new, defaulting to rigid; ungrouped faces form a default walls group.
  - It never resends existing boundaries.
  - Roles, weights, components, excitations and channels belong to the user and survive regeneration through the host rule that omitted fields are preserved.
- **Re-import matching:** faces are matched by a unique STEP name, then by fingerprint within a tolerance (among the faces sharing the STEP name, if several do): same surface type, centre within 1 % of the bounding-box diagonal, area within 5 %, normal pointing the same way within about 2.6° (a flipped normal does not match). A stored face whose fingerprint matches more than one face is ambiguous and counts as not found. If any face group is missing any of its stored faces (none or only some found, so nothing is reassigned silently), generation is rejected before the response is sent with one message naming every such group; the previous meshes stay in place. The editor runs the same matching on the loaded file and lists those groups as needing review. Grouping a selection under the name of a group that needs review drops its missing faces, which repairs it. Deleting a group, like any face group that disappears, sends on the next generation the explicit removal of its boundary, of interfaces using it, and its detachment from its component in the same patch; a component left without boundaries is removed with its excitations. A component that also holds boundaries of another design is left as is, so the host rejects the generation until the user detaches the boundary in System. Bounded regions whose mesh is gone are removed too.
- **Watching the STEP file:** when the application regains focus, a changed STEP file (modification time or size) triggers a re-mesh prompt, once per change, rather than an automatic regeneration. The file is stamped before it is read, so an edit while it loads still prompts; a design without a STEP file never prompts. Yes reloads the preview and starts the host's Generate.
- **Symmetry suggestion:** open edges lying entirely on X=0 or Y=0 after reorientation suggest the matching project symmetry. A mismatch with the project setting rejects generation with a message.
- **Editor:** a 3D viewport with face picking above a body and face tree, with visibility checkboxes, Show all, and Clear (which shows everything and clears the selection, never deleting groups). A click in the view selects a face, Ctrl or Shift+click adds or removes one; the selection is shared with the tree and highlighted in the view. The tree has a size override column for bodies. Below that sit the face-group list (name field, Group selection, which creates the group or adds to an existing one, Delete group, and a size override column; groups named in the STEP file are listed too, and selecting a row selects its faces), the orientation pickers, and the sizing settings with the computed target element size. Meshing uses the host's Generate (F7) and geometry-generation cancel. When a loaded STEP file has closed shells without a stored choice, the editor asks for each one (Air volume / Exterior surface; Escape means exterior) and stores the answers; a Closed shells list keeps each choice editable afterwards. Choosing another STEP file clears the stored choices. The editor follows host revisions and busy state like the vented-enclosure editor.
- **CLI:** a thin command over the same headless pipeline taking a STEP path, the maximum frequency, elements per wavelength and an output folder, and writing the assembly meshes. A repeatable `--closed-shell NAME=air|exterior` sets closed shell choices.
- **Licence:** Fusion2Msh (CC BY-NC 4.0) informed the design only. Its code must not be copied into Boundary Lab, which is GPL-3.

## Testing Decisions

- **Good tests check external behaviour only:** the meshes produced and their physical groups, edge lengths, conformity at interfaces, the System entries that compile, rejections and their messages, and user edits surviving regeneration. No tests on internal helpers.
- **Primary seam: the provider through the public generator contract.**
  - Load the plugin through the provider catalog, generate from a STEP fixture and design source, stage the result, then compile the physical system or prepare a solve.
  - Prior art: the vented-enclosure generator tests.
  - Covers sizing (through the edge lengths of generated meshes), interface detection, axis mapping, symmetry detection, System entries, and re-import matching (regenerating from a modified fixture).
- **Slice 0** uses the same contract with the existing test assembly fixture: an already-reduced variant accepted under X and XY symmetry, a resource outside the positive region rejected atomically, and unreduced assemblies still rejected. Prior art: the generator assembly tests.
- **UI seam:** the main window with the CAD Import dock. Generate from the dock, check the visibility tree and Show all / Clear, and drive face picking through the editor's API rather than mouse clicks. Prior art: the vented-enclosure UI tests.
- **CLI:** a thin end-to-end run on a fixture. Prior art: the CLI tests.
- **Fixtures:**
  - SolidWorks AP214 exports in millimetres of a vented box (chamber, port, exterior) and a waveguide (throat volume, exterior), committed if small.
  - A test helper strips names to exercise matching by fingerprint.
  - Until the files exist, geometry is built in code with Gmsh's OpenCASCADE kernel and written to STEP.

## Out of Scope

- Small-feature or edge-proximity refinement (deferred to v2).
- Free-angle rotation (only 90° axis mappings).
- Moving the origin inside CAD Import (the CAD origin is used; Meshes translation remains available).
- Host cutting of full assemblies under symmetry; only assemblies that are already reduced are accepted.
- Interfaces between two FEM regions; touching air bodies are merged instead.
- Treating solids as cabinet material and deriving the air by subtraction.
- Assigning boundary roles or components inside CAD Import; that stays in System.
- Changing the generator-provider contract beyond the symmetry declaration.

## Further Notes

- **Delivery:** five slices on one feature branch, opened as separate PRs once the whole feature is built:
  0. Host symmetry change.
  1. Headless core and CLI.
  2. Provider package, 3D editor and visibility tree.
  3. Face picking, face groups, orientation pickers and System entries.
  4. Re-import matching and watching the STEP file.
- **Vocabulary** follows the root glossary: air body, exterior surface body, face group, target element size, CAD import. A "source" is a moving boundary owned by a component.
