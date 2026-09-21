# Geometry provider API: generation and configuration

This guide covers the implemented provider API (exchange schema version `1`):
generation, configuration, in-memory geometry, host solve/result services,
local package discovery, and custom document editors.

## Implemented boundary

```text
Host document + configuration snapshot
  -> GenerationRequest
  -> GeneratorBackend.create_session(request)
  -> session.generate(status_callback=..., stop_requested=...)
  -> GenerationResponse
  -> validate identity and patch syntax
  -> reject stale/replayed responses
  -> stage geometry reference and configuration on a detached project
  -> validate ownership and references
  -> commit project and geometry together
  -> refresh mesh preview
```

The generation backend and configuration/application modules do not depend on
Qt. The desktop worker adapts their results to Qt signals. The physical compiler
and normal solve preparation still decide whether a model is solve-ready;
acceptance does not assert that every driver parameter or numerical assumption
is complete.

### Current scope

- `GeneratedGeometry` accepts one immutable `MeshData` snapshot or one mesh file
  per design, with optional symmetry variants. Memory snapshots pass through
  preview, compilation, exterior stitching and the local BEAT worker without
  mesh/request-file round trips. File providers must write distinct artifacts
  and never overwrite a file belonging to an accepted revision.
- Physical edits use the existing physical model. Mesh resources are supplied
  by the host's accepted artifact, not by configuration patches.
- A first exterior generation can seed the standard exterior model from mesh
  groups and legacy radiator hints. Existing FEM/coupled models can be edited
  within the provider's ownership scope. Creating a new multi-mesh FEM/coupled
  contribution is not implemented in this increment.
- Local folders with a `provider.json` manifest appear in Preferences → Geometry
  providers → Manage packages. Enable a package before its code can be loaded.
  Custom widgets live inside the host's design dock; Ath remains the default.
- Providers can submit asynchronous solves, subscribe to status, cancel their
  own jobs, and query canonical solved data through the host services below.
  The host owns the frequency range and count; provider commands cannot change them.
- Preview/solve preparation remains serialized. Memory transport uses packed
  buffers inside JSON; shared memory and concurrent preview/solve preparation
  remain future work. No reduced solve latency is claimed.

## Request contract

Import the public exchange types from `blab.generators`.

| Field | Owner / meaning |
| --- | --- |
| `schema_version` | Host exchange version; currently `1` |
| `request_id` | Host-issued unique ID; echo it exactly |
| `provider_id` | Provider selected by the design |
| `document_id` | Stable design identity |
| `mesh_name` | Current host mesh name for this design |
| `source` | Detached snapshot of the provider's saved source dictionary |
| `project_revision` | Opaque host revision token covering source, settings and artifact references |
| `configuration` | Detached physical-system and configuration snapshot |
| `run_root`, `case_name` | Existing file-generation output context |
| `provider_options` | Host-provided runtime options, such as the Ath executable |

`configuration` contains `physical_system`, `channel_config`,
`component_channels`, `symmetry_config`, and `stitching_config`. Inspect the
physical system to discover existing mesh, boundary and component IDs. Do not
derive IDs from display labels. It may be `None` before the first generation;
generate a mesh first to obtain host-assigned IDs on the next request.

Provider code must treat the request as read-only. Editing its detached
dictionaries does not edit the host project. The revision token currently uses
a content digest; it is not a monotonic edit counter and does not hash mesh file
contents. Selecting another design tab alone does not invalidate a request.

## Response contract

```python
from blab.generators import GenerationResponse

def generate(self, *, status_callback=None, stop_requested=None):
    # self.request was supplied by create_session().
    # build_geometry returns the existing GeneratedGeometry artifact type.
    geometry = self.build_geometry(status_callback, stop_requested)
    return GenerationResponse(
        schema_version=1,
        request_id=self.request.request_id,
        geometry=geometry,
        configuration_patch={
            "channel_config": {
                "upsert": [{"name": "main", "level_db": -3.0}]
            }
        },
    )
```

The worker rejects unknown versions, mismatched request IDs, mismatched provider
IDs, malformed patches, unknown fields, non-finite JSON values and conflicting
operations. The response's metadata and patch are detached before publication.
The host rejects completions for obsolete requests, changed projects or removed
designs; one request can be accepted only once by the desktop workflow.

Ath and existing backends returning bare `GeneratedGeometry` are adapted to a
correlated completion and retain their legacy exterior-seeding behavior. New
providers should return `GenerationResponse`, including when they have no
configuration changes, to use transactional configuration acceptance.

## Patch semantics

1. An omitted section or field preserves its current value.
2. Nested objects merge recursively; supplied arrays replace that array.
3. Empty objects and empty operation arrays make no changes.
4. `null` is not an erase command. Use explicit entity `remove` operations.
5. Removing an entity requires resolving its dependent references in the same
   patch. Removed components' channel routes are removed with them.
6. New physical entities need their required authoring fields. Missing physical
   parameters are not invented; normal solve validation reports incomplete setup.
7. A malformed or conflicting patch leaves the accepted project and artifact
   references untouched. Generated output files are not deleted on rejection.

Frequency settings, backend settings, arbitrary project preferences and mesh
file paths are not accepted patch fields. Channel crossover frequencies are
channel parameters and do not change the solve's frequency sweep.

### Entity operations

These sections use `{"upsert": [...], "remove": [...]}`. Both arrays are optional.

| Section | Entity fields / model |
| --- | --- |
| `regions` | `AcousticRegion`: ID, name, kind, mesh IDs, volume groups, fluid properties and loss model |
| `surface_assignments` | `Boundary`: ID, name, region ID, group reference, kind, parameters |
| `interfaces` | `AcousticInterface`: ID, name, two boundary IDs, coordinate tolerance |
| `component_assignments` | `PhysicalComponent`: ID, name, kind, boundary IDs, parameters |
| `excitation_ports` | `ExcitationPort`: ID, name, component ID, kind |
| `channel_config` | `ChannelConfig`, keyed by **name** instead of ID |
| `component_channels` | `{"id": component_id, "channel": channel_name}` |

The physical field definitions and enums are in
[`physical_model.py`](../../src/blab/physical_model.py). JSON enum values are
strings, for example `"moving"`, `"ideal_velocity_source"`, or `"normal_velocity"`.

Existing physical entities can be modified only when they belong entirely to
the generating design's mesh resources. New entity IDs must start with
`document_id + "/"`, for example `design-7/driver`. References must still point
inside that scope. A shared exterior region can be referenced, but a provider
owning only some of its meshes cannot rewrite that region. Cross-provider
interface authoring will need an explicit host composition API.

Channel, symmetry and stitching sections explicitly request project-wide
changes. They are not inferred from provider-specific metadata. Removing an
in-use channel requires rerouting its components, and at least one channel must
remain. Removing a component route restores the host's `main` fallback, which
must exist for an excited component.

Example using IDs obtained from the request snapshot:

```python
patch = {
    "surface_assignments": {
        "upsert": [{"id": throat_boundary_id, "kind": "moving"}]
    },
    "component_parameters": {
        driver_component_id: {"re_ohm": 6.2}
    },
    "channel_config": {
        "upsert": [{"name": "HF", "level_db": -2.0}]
    },
    "component_channels": {
        "upsert": [{"id": driver_component_id, "channel": "HF"}]
    },
    "symmetry_config": {"mode": "x"},
    "stitching_config": {"enabled": True, "tolerance_mm": 0.25},
}
```

`component_parameters` is a mapping from component ID to a parameter-object
patch. It merges after parameters supplied through `component_assignments`.
Individual parameter-key deletion is not implemented; remove/recreate an entity
in separate accepted updates or use the application's authoring UI when a kind
change requires removing incompatible parameters.

Symmetry modes currently match the host: `off`, `x`, `xy`. This patch selects the
host setting; it does not certify that a mesh has the requested symmetry. Normal
mesh/solver validation still applies. Stitching currently maps to the host's
existing project-wide stitching switch and nonnegative tolerance in millimeters;
per-pair stitching policies are not implemented.

## Regeneration and persistence

For versioned responses, the host stages the new mesh source and merges the
patch into a detached project. It checks that preserved physical-group
references still exist on the generated mesh. When both name and tag are
specified, both must match, as required by the physical compiler. Use name-only
references when tags are expected to change during remeshing.

Accepted configuration is ordinary project state and uses the existing
`.blab.json` persistence path. Applied physical edits disable legacy automatic
exterior reseeding so a later solve cannot replace those edits with radiator
defaults. Headless consumers can call `complete_generation()` and
`stage_generation()` directly; they must commit the candidate and its geometry
together and enforce request ordering themselves.

In-memory artifacts are embedded in `.blab.json` on explicit project save and can
be restored without importing the provider. File artifacts still use
`backend.restore()`. A provider is required to generate again. Multi-mesh provider
contributions remain a separate increment.

## Installing and developing provider packages

Copy a package folder into `<Boundary Lab>/geometry_providers/`, the application's
only provider location. Preferences → Geometry providers → Manage packages offers
an **Open install folder** shortcut and **Rescan**. Enable the
package, accept Preferences, then choose it as the **default geometry provider**
for new designs. Existing designs keep their own provider ID. Newly discovered
packages are disabled until enabled; opening a project never enables a package.

The runnable [Example Box package](../../examples/geometry_providers/example_box/)
contains a custom Qt editor and a backend producing a closed mesh in memory.
Copy its `example_box` folder, enable **Example Box**, select it as the default,
and add a design. Edit the dimensions and click the host's **Generate** button.
Its twelve triangles demonstrate the integration rather than acoustic accuracy.

```text
example_box/
  provider.json
  box/
    __init__.py
    backend.py
    editor.py
```

```json
{
  "manifest_version": 1,
  "id": "example.box",
  "name": "Example Box",
  "version": "0.1.0",
  "provider_api": 1,
  "source_schema_version": 1,
  "backend": "box.backend:create_backend",
  "editor": "box.editor:create_editor",
  "default_source": {"width_m": 0.1, "height_m": 0.1, "depth_m": 0.1},
  "mesh_scale_factor": 1.0
}
```

| Manifest field | Contract |
| --- | --- |
| `manifest_version`, `provider_api` | Required integer `1`; incompatible versions cannot load |
| `id` | Stable lowercase ID; duplicate IDs, including `ath`, are conflicts |
| `name`, `version` | Required display strings |
| `source_schema_version` | Required positive integer; must equal the saved design schema |
| `backend` | Required `module:factory`; factory accepts host runtime keyword options and returns a backend |
| `editor` | Optional `module:factory`; called on the GUI thread as `factory(parent, document_host)` |
| `default_source` | Optional JSON object, defaults to `{}`; copied into new designs |
| `mesh_scale_factor` | Optional positive units-to-metres conversion, defaults to `1.0` |

Discovery reads manifests and fingerprints Python sources without importing
them. Only enabled packages load code, lazily. Backend factories run in generation
workers; editor factories run on the GUI thread. Do not import Qt from your backend
or access widgets during generation. Use relative imports for private modules:
packages receive a private module namespace, without additions to `sys.path`.

Providers are trusted in-process Python code, with the application's filesystem
and process access. This is not a sandbox. Use host dependencies or package-private
Python modules; the host does not run pip, create environments, or install native
dependencies. Binary extensions must match the host's Python and platform.
Loaded Python code is retained for the process lifetime. After modifying or
replacing a loaded package, restart Boundary Lab; rescan is not hot reload.

### Editor and document contract

Return an adapter implementing `blab.ui.provider_editor.ProviderEditor`:

```python
class WaveguideEditor:
    widget: QWidget  # your own widget, not a QDockWidget

    def apply_source(self, source: dict, revision: str) -> None:
        # Populate controls without emitting edits back to the host.
        ...

    def set_operation_state(self, state) -> None:
        # state.active, state.phase, state.message; keep this callback short.
        ...

    def dispose(self) -> None:
        # Disconnect your signals; stop your timers and private workers.
        ...
```

Boundary Lab owns dock placement, tabs, Generate/Stop, project persistence, and
widget deletion. Its dock retains the existing `ath_editor_dock` layout identity.
Ath uses this same adapter lifecycle. Each rebuild disposes old editors and
creates new ones; keep persistent parameters in source, not just in widget state.

The supplied `DocumentHost` exposes:

| Method/property | Meaning |
| --- | --- |
| `document_id`, `provider_id` | This editor's document identity |
| `snapshot()` | Detached `SourceSnapshot(source, revision, schema_version)` |
| `update_source(source, expected_revision=...)` | Replace the whole source JSON object; return a fresh snapshot |
| `generate(expected_revision=...)` | Select this document, start normal generation, return the request ID |
| `services` | Scoped `ProviderHost`: context, solve, status, cancel, result, current_result, subscribe, unsubscribe |

Source methods and `generate` are synchronous and GUI-thread only. Source updates
are persisted immediately in the project model and participate in unsaved-change
tracking, before generation. Retain unknown source fields when updating controls.
Revisions are opaque content tokens: stale updates and commands raise
`ProviderHostError`; read a fresh snapshot and reconcile controls. This source
revision is distinct from the project revision required by `SolveCommand`.

```python
snapshot = document_host.snapshot()
source = snapshot.source | {"width_m": width_spin.value()}
snapshot = document_host.update_source(source, expected_revision=snapshot.revision)
request_id = document_host.generate(expected_revision=snapshot.revision)
```

`generate` rejects busy hosts; it does not queue interactive mesh requests. Coalesce
slider edits until release. Source edits made during generation invalidate its
captured revision, so the host rejects stale geometry on return. Normal editors
disable controls while `state.active` is true.

`services` retains the asynchronous Future contract described below. Its event
callbacks execute on the host GUI thread. An editor can subscribe to
`geometry_accepted`, then submit `SolveCommand(event.context.project_revision)`
through `document_host.services.solve(...)`. Match the generation request ID if
only a particular generation should trigger a solve. Wait for acceptance, not
just completion of `generate()`, which only starts the worker. Never block the
GUI thread waiting on a host Future.

Editor subscriptions are automatically removed on disposal, including pending
subscription acknowledgements; late events cannot call the disposed editor.
Disposed document handles reject new calls. New/open project operations invalidate
old handles even when document IDs happen to match.

Missing/disabled/incompatible providers and editor construction failures show a
read-only placeholder with the original source. Saved memory artifacts restore
without provider code. No automatic source-schema migration is performed; an
incompatible schema disables generation until a compatible provider is installed.

## Explicit registration during development

```python
from blab.generators.base import GeneratorCapabilities
from blab.generators.registry import GeneratorBackendInfo, register_generator

register_generator(GeneratorBackendInfo(
    provider_id="example.waveguide",
    label="Example waveguide",
    capabilities=GeneratorCapabilities(source_formats=("example_parameters",)),
    factory=WaveguideBackend,
))
```

The backend implements `create_session(request)` and `restore(document)`. Sessions
implement `generate(...)` and `stop()`. Registration cannot overwrite an existing
provider, including Ath. Development code must explicitly register the backend
before loading/generating its documents. Project loading does not import code
from paths supplied by a project.

Explicit registration remains useful for tests and embedded hosts. Distributable
providers should use the manifest contract below. Providers without an editor
display preserved source as read-only JSON and can still generate.

## Validation

Focused tests cover the contract, worker boundary, desktop acceptance, stale
completion handling, ownership, explicit removals, nested preservation, frequency
ownership, and rejection without partial mutation:

```sh
python -m pytest tests/test_provider_api.py tests/test_geometry_workflow_controller.py tests/test_generator_backends.py
python -m ruff check src tests
```

## In-memory geometry

```python
from blab.generators import GeneratedGeometry, GenerationResponse, MeshData

mesh = MeshData(
    points=vertices,                         # (N, 3), finite coordinates
    cells=[("triangle", triangles)],         # (M, 3), zero-based integer indices
    physical_tags=[surface_tags],            # (M,), positive integer tag per cell
    physical_names={"wall": [1, 2], "throat": [2, 2]},  # name: [tag, dimension]
)
return GenerationResponse(
    request_id=request.request_id,
    geometry=GeneratedGeometry(
        provider_id=request.provider_id,
        output_dir=request.run_root,        # retained for compatibility; no mkdir needed
        mesh_path=None,
        radiators=(),
        mesh_data=mesh,
    ),
    configuration_patch=configuration_patch,
)
```

`MeshData.from_meshio(mesh)` is also available. Construction detaches input
arrays; published arrays are read-only. Preview/preparation obtains mutable
working copies, so providers can reuse their original arrays for another
generation. Physical groups must name every used cell tag. Cell blocks retain
their order. Supported types are `triangle`, `triangle6`, `tetra`, and `tetra10`,
in meshio/VTK ordering. BEM requires linear triangles; quadratic FEM requires
matching quadratic volume and surface elements. CAD entity tags and other
meshio metadata are outside this acoustic contract; preparation synthesizes
entity labels from physical groups for its legacy conformer.

Coordinates use the document's `mesh_scale_factor` and `mesh_translation_mm`;
set scale to `1.0` for meters (the existing default is `0.001`). Winding and
physical-group identity have the same meaning as file meshes. Omitted physics
continues to preserve host configuration. Frequencies remain host-owned.

Set `mesh_path=None` for memory geometry. Resources and compiled meshes use
`file=""` plus `mesh_data`; supplying both is rejected. If full geometry declares
`mirror_axes`, supply `reduced_mesh_data` before enabling native symmetry. The
host does not silently cut an in-memory mesh to a symmetry domain.

### Worker transport and limits

The local worker advertises `contracts.mesh_data: [1]` and
`request_transports: ["file", "inline_json"]`. The application sends the solve
request as one JSON command containing packed little-endian float64 coordinates
and int64 connectivity/tags, base64 encoded. Older workers reject this path
before accepting the job. This requires the updated BEAT runtime; existing file
projects continue using their established transport.

This avoids mesh and request files, but still incurs buffer copies, base64 and
JSON encoding. It is not shared memory and makes no latency guarantee. Cancelling
an inline solve currently terminates that worker; the next solve starts another.
Remote asset transport and legacy simulation transport reject memory meshes.
Exact Level-3 package export still requires file assets. Explicit project/result
saving can write to disk as usual.

Solve and preview can consume the same immutable snapshot independently. Host
services below support solve requests after generation. Provider widgets can
commit a slider value and request generation on release. Concurrent preview and
solve preparation remains separate work.

```sh
python -m pytest tests/test_memory_mesh.py tests/test_exterior_preparation.py
# Optional real CPU + CUDA parity: set BLAB_TEST_MEMORY_SOLVE=1,
# then run tests/test_memory_mesh.py -k real
```

## Host services: solve, status, and results

The desktop injects a document-scoped `ProviderHost` into backends implementing
the optional `bind_host(host)` method, before `create_session(request)`. Ath and
backends without this method keep their existing behavior. Services are not
serialized into project files or passed as generation configuration.

Every method returns a `concurrent.futures.Future`. Calls can originate on worker
threads; the desktop dispatches application access onto its GUI thread. A solve
Future acknowledges a **queued job**, not completion. Event and Future callbacks
run on the dispatch thread: keep them short, move expensive optimization to a
worker, and never block there waiting for another unfinished host Future.
Subscriber exceptions are logged without stopping the host.

| Method | Future result | Meaning |
| --- | --- | --- |
| `context()` | `ProviderContext` | Document/provider IDs, revision, host frequency minimum/maximum/count |
| `solve(SolveCommand(...))` | `SolveJob` | Queue a solve for that exact revision |
| `status(job_id)` | `SolveJob` | State, progress counts, request ID, revision, result run ID |
| `cancel(job_id)` | `SolveJob` | Cancel this document's pending/preparing/running job |
| `result(job_id)` | `SolvedSystem` | Detached complete or partial snapshot after the job settles |
| `current_result()` | `SolvedSystem` | Host's latest finalized snapshot, including manual solves |
| `subscribe(callback)` | Subscription ID | Receive this document's generation acceptance and solve status events |
| `unsubscribe(subscription_id)` | `None` | Release a callback when its consumer is disposed |

### Requests and scheduling

```python
from blab.generators import SolveCommand

# On a provider worker thread; never block a GUI/event callback this way:
context = host.context().result(timeout=5)
job = host.solve(SolveCommand(
    expected_revision=context.project_revision,
    replace_pending=True,
)).result(timeout=5)
```

`SolveCommand` contains only `schema_version=1`, `request_id` (a generated UUID by
default), `expected_revision`, and `replace_pending=False`. It cannot override
frequencies, backends, outputs, or physics. Solving uses the same preparation,
streaming result builder, and plots as the application's Solve button. An accepted,
solve-ready physical system is required. Exterior topology warnings fail the
provider job instead of opening an automated confirmation dialog.

The solve revision includes project content, current frequency settings, and
application solver preferences. Obtain it through `host.context()` or a
`geometry_accepted` event; the generation request's revision is a different token.
Changed inputs reject submission (`stale_revision`) or discard queued work
(`stale`). Changes during preparation discard that preparation and fail the job.
External file-content edits are not tracked by this token: use immutable memory
snapshots or distinct file artifacts for each accepted generation.

One provider request can wait alongside active host work. `replace_pending=True`
supersedes only a pending request from the same provider document; it never
interrupts an active solve. Another document cannot replace or cancel that job.
Pending requests wait for generation, preview preparation, or an existing solve
to settle, then recheck their revision. This bounds the queue; it does not add
parallel preparation or remove cold worker startup costs.

States are `queued`, `preparing`, `running`, `cancelling`, then one of `completed`,
`failed`, `cancelled`, `superseded`, or `stale`. Completion requires a full
frequency result set. Cancellation during preparation suppresses publication;
the calculation may finish in the background. Numerical cancellation uses the
backend's existing behavior (inline BEAT currently retires its worker).

Identical retries return the same job while its history is retained. Reusing an
ID with different contents gives `request_conflict`; use a fresh ID for a new
solve. The host retains four result snapshots and up to 32 job records. Lookup
after eviction gives `result_unavailable`, or `unknown_job` when the job record
has expired. These are in-session handles, not durable project identifiers.

### Solve after generation is accepted

The host publishes `ProviderEvent(kind="geometry_accepted", context=...,
generation_request_id=...)` only after committing geometry and configuration.
The context contains the new revision. For example, a retained provider consumer
can react without blocking the GUI:

```python
def on_host_event(event):
    if event.kind == "geometry_accepted":
        command = SolveCommand(expected_revision=event.context.project_revision,
                               replace_pending=True)
        host.solve(command).add_done_callback(on_submission)
    elif event.kind == "solve_status":
        job = event.job
        print(job.request_id, job.state, job.solved_count, job.expected_count)
        if job.state == "completed":
            host.result(job.job_id).add_done_callback(on_result)

def on_submission(future):
    job = future.result()  # already done in this callback; handle errors here
    print("Queued", job.job_id)

def on_result(future):
    solved = future.result()  # already done; send heavy processing to a worker
    print(solved.run_id, solved.complete, tuple(solved.quantities))

# Set up on a worker thread; keep and eventually release the subscription ID.
subscription_id = host.subscribe(on_host_event).result(timeout=5)
# On consumer disposal: host.unsubscribe(subscription_id)
```

Register once per retained consumer. A generation backend that subscribes in
`bind_host` must unsubscribe when its consumer/session is disposed, including
failed or cancelled generation. Filter `generation_request_id` and
`job.request_id` when multiple consumers share a document; events cover that
whole document. There is a limit of 64 live subscriptions per document binding.

New/open project and document removal revoke old handles, drop their
subscriptions/results, and cancel their work. Accepted regeneration within the
same project keeps handles valid. Providers are trusted local Python code;
document scoping is an API ownership rule, not a process sandbox.

### Result queries and errors

`SolvedSystem` is the same canonical model the host uses: complex values per
frequency/excitation, units/dimensions, domain coordinates/topology, availability
masks, diagnostics, compiled system, and provenance. Enumerate `solved.quantities`
or use `solved.quantity(quantity_id)`. Returned copies cannot mutate host results.
Derived helpers in `blab.solve_results` cover SPL, phase, electrical impedance,
and velocity-to-excursion conversion. Channel mixing and presentation remain
distinct from the raw excitation basis.

`current_result()` returns the latest finalized complete or partial host run;
it does not assert those results match newly edited geometry. Use `result(job_id)`
to correlate an optimizer evaluation with its request. This API does not expose
live numerical arrays, pressure probes requiring additional engine work,
host-driven generation, or custom provider widgets yet.

`ProviderHostError.code` includes `busy`, `stale_revision`, `request_conflict`,
`invalid_context`, `unknown_job`, `not_ready`, `result_unavailable`,
`subscription_limit`, and `closed`. Command/status/event schemas are version 1.
Failures after acceptance appear as terminal job events with a message, not as
exceptions on the already acknowledged solve Future.

```sh
python -m pytest tests/test_provider_host.py tests/test_ui_provider_host.py tests/test_solve_workflow_controller.py
python -m ruff check src tests
```
