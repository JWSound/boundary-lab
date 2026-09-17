# Geometry provider API: generation and configuration

This is the first implemented increment of the provider API (exchange schema
version `1`). It connects a versioned response to the existing Generate workflow
and project configuration. It is not yet the complete plugin SDK.

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

- Geometry transport is still `GeneratedGeometry`: one mesh file per design,
  with optional cleaned and reduced variants. No array-to-temporary-file adapter
  has been introduced. Providers must write distinct output artifacts and never
  overwrite a file belonging to an accepted revision.
- Physical edits use the existing physical model. Mesh resources are supplied
  by the host's artifact path, not by configuration patches.
- A first exterior generation can seed the standard exterior model from mesh
  groups and legacy radiator hints. Existing FEM/coupled models can be edited
  within the provider's ownership scope. Creating a new multi-mesh FEM/coupled
  contribution is not implemented in this increment.
- Registration is explicit Python registration; folder discovery, manifests,
  installation preferences and provider editor widgets remain future work.
- Provider solve submission, solve status subscriptions, and solved-system
  queries are not implemented yet. The host continues to own the frequency
  range and count; this configuration API cannot change them.
- Concurrent preview/solve scheduling and binary in-memory engine transport
  remain future work. This increment does not claim reduced solve latency.

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

For versioned responses, the host stages the new file reference and merges the
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

GUI restoration of provider-specific artifacts still uses `backend.restore()`.
Provider-independent restoration and normalized multi-mesh artifact persistence
remain part of the next transport/persistence increment.

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

The current UI displays non-Ath source as read-only JSON. Use a programmatically
created `GeneratorDocument` or project file to exercise a registered backend
until custom editor adapters are available.

## Validation

Focused tests cover the contract, worker boundary, desktop acceptance, stale
completion handling, ownership, explicit removals, nested preservation, frequency
ownership, and rejection without partial mutation:

```sh
python -m pytest tests/test_provider_api.py tests/test_geometry_workflow_controller.py tests/test_generator_backends.py
python -m ruff check src tests
```
