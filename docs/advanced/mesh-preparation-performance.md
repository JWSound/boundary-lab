# Mesh preparation and caching

Desktop previews and System inventory inspection share the Qt-free
`blab.mesh_cache` and `blab.mesh_inventory` contracts. Exterior preparation and
topology analysis use the same cached reader, including headless workflows.

The process-local LRU cache retains up to 32 meshes with a 128 MiB estimated
storage budget. The estimate includes array sizes and per-block headroom; this
is not a hard process-memory limit. Large meshes exceeding the budget are read
without retention. Each key includes the resolved source path, size, modification
and creation/change timestamps, and filesystem identity. A changed file replaces
its earlier entry. The reader checks the fingerprint again after parsing and
rejects a file that changed during the read. Missing files still raise errors.

`read_mesh(path)` returns an independent deep copy. Callers may transform points,
edit physical groups, and construct interfaces without corrupting subsequent
reads. `mesh_cache.inventory(path)` returns immutable physical-group inventory,
including surface ownership for each FEM volume. The inventory shares the raw
mesh entry and its invalidation lifetime.

File reloads explicitly invalidate source and cleaned paths. Generated-file
replacement is detected through the same fingerprints. Call
`mesh_cache.invalidate(path)` when an operation deliberately changes content
while preserving filesystem metadata. `invalidate()` clears the whole cache.
The cache does not stage or convert the user's original files.

Canonical and symmetry inventories reuse matching individual mesh entries.
Changing one generated waveguide no longer causes unchanged imported meshes to
be inspected twice. Scale and translation remain properties of each inventory
entry; raw geometry and volume ownership are independent of those transforms.

An authored physical system does not restore legacy radiator assignments during
project loading. Its loader skips the redundant `surface_tags_for_meshes()`
assembly and prepares the preview once. Legacy/auto-seeded source migration
retains its previous behavior for compatibility.

`MeshPreparationSnapshot` captures document, geometry, imported-mesh, physical
system, symmetry and stitch settings. A worker materializes generated symmetry
meshes, prepares the assembly, computes topology and hierarchy, and returns mesh
data with the preview options. VTK consumes that data on the GUI thread without
opening mesh files again. Workers do not mutate the active project. Callbacks
reject obsolete project identities and re-prepare if their snapshot changed.
Starting a new project cannot receive a previous project's late preview.

System opening inspects its snapshot in a worker and constructs the dialog on
the GUI thread after the job completes. Project file reading and generator
restoration also run in a worker. Solve/generation and mesh-editing controls are
locked while preparation is active. See [activity integration](ui-activities.md)
for cancellation and lifecycle behavior.

## SAWMAX measurements, 2026-09-12

The isolated Qt harness uses the actual main-window workflows with the test
preview widget in place of VTK, and excludes user interaction with dialogs.
The application event loop runs normally while workers prepare geometry.
Both comparisons retain Google Drive source paths and use the same default
application preferences (including 2 mm stitching; the saved project specifies
3 mm). Prior timings include cProfile overhead; these comparisons are indicative,
not a rendering benchmark or a claim of a precise speedup factor.

| Operation | Previous profile | Worker/cache measurement |
| --- | ---: | ---: |
| Initial project load | 32.37 s | 5.62 s |
| First System opening after load | 8.22 s | 0.154 s |
| Repeated System opening | 8.20 s | 0.032 s |
| Warm project reload | — | 1.26 s |

Initial loading made five mesh parses instead of 21, including only one SideSlot
parse. Every measured parse occurred off the GUI thread. First System opening
read only the canonical generated waveguide; repeat System opening and warm
project reload made no mesh parses. Cold action dispatch returned in under 1 ms.

A 20 ms heartbeat observed a largest event-loop gap of approximately 0.69 s
during cold loading and 1.02 s during warm loading. UI state updates and plot
clearing remain work to investigate; actual VTK rendering is excluded here.
The first Google Drive mesh parse also still takes several seconds, even though
it no longer blocks the GUI thread.

SAWMAX headless validation passed after the changes. Its prepared exterior mesh
retained SHA-256 `799d8cdcee863ebf4c9562a547c3928f564048327f2e8b5a175175d35d53ca2f`,
matching the earlier investigation. No acoustic solve was run. Local diagnostic
scripts and raw timings are retained in ignored `runs/performance_investigation`.

## Solve preparation

Normal exterior and coupled/FEM Solve actions snapshot the project, preferences,
frequency range and mesh entries, then inspect inventories and construct the
request in the shared preparation worker. Completion publishes on the GUI thread
only if those inputs are still current. Mesh/project refresh cancels pending solve
preparation. Stop suppresses completion and keeps controls locked until running
preparation returns. Previous results survive until the new run begins.

The activity indicator starts with “Preparing solve…” and reports mesh inspection,
exterior interface preparation, compilation, and result-domain construction.
Queued signals deliver progress; stale progress is ignored. Core preparation
accepts an optional plain callback and remains independent of Qt. Per-frequency
completion/timing status is unchanged.

Physical compilation, component geometry and FEM result-domain construction now
use the shared mesh reader. Entries remain fingerprinted, bounded and protected
against caller mutation. Interface validation is retained. Speaker-package export
uses these cached reads but retains its separate synchronous preparation workflow.

For the larger SAWMAX project with its saved 3 mm stitch tolerance, the former
warm click-to-worker-handoff delay was 10.34 s, including about 8.05 s of repeated
mesh parsing. Two background runs reached handoff in 2.50 s and 3.09 s; click
dispatch returned in approximately 1 ms. The first run parsed only the prepared
exterior (8 ms), and the repeated run performed no mesh parses. The GUI continued
processing a 20 ms timer in both runs. Compiled systems matched across runs.

These are individual measurements using the normal Qt event loop, a stub preview,
and no previous solved dataset. Plot clearing still runs on the GUI thread; this
is not a full rendering benchmark. No acoustic solve was launched. Raw results
are in ignored `runs/performance_investigation/solve_background_results.json`.
