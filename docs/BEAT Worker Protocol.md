# BEAT worker protocol v1

The persistent physical-system worker announces its capabilities before accepting
a job. The client selects supported wire formats in each command. Compatibility
is checked before the client writes a command, and Julia checks that selection
before opening the request file. No matrix assembly is needed for negotiation.

## Handshake and version ownership

[worker-v1.json](../src/blab/solvers/beat_contract/worker-v1.json) is the engine-owned
announcement template. Julia adds `runtime.julia_version` and `backends` at startup,
then emits the resulting object as a single JSON line on stdout, with `type: ready`.

| Field | Meaning |
|---|---|
| `protocol` | Name `beat-worker`, integer version `1` |
| `engine` | Name `BEAT Engine`, development version `0.1.0-dev` |
| `contracts` | Supported versions for system request, compiled system, system result, and field array |
| `operations` | `solve` and `bem_field` |
| `precisions` | `float32` and `float64` |
| `solve_kinds` | `exterior_bem`, `interior_fem`, `coupled_fem_bem_lem` |
| `backends` | Per-backend `available` boolean and optional explanatory `reason` |
| `cancellation` | `marker_file`, for physical solves |
| `field_cache` | Process lifetime, clearing conditions, and maximum entry count |
| `runtime` | Julia runtime version |

The development engine version identifies this engine interface baseline; it is
independent of Boundary Lab's application version and is not a Git revision or
reproducible-build identifier. Compatibility uses protocol and contract versions,
not comparisons of engine version strings. The announcement also carries
[engine revision, source hashes, and runtime provenance](BEAT%20Run%20Provenance.md).

CPU is available in the bundled worker. CUDA is advertised available only when
its loaded module reports a functional runtime/device. ROCm additionally requires
functional rocBLAS and rocSOLVER. Probe failures are reported as unavailable with
a reason. These are runtime availability checks, not numerical qualification or
guarantees that a particular problem fits device memory. Availability is sampled
for each new process; subsequent device failures are normal job errors.

Unknown additive handshake fields are allowed. Protocol version must be an integer
equal to 1; booleans, strings, and floating-point tokens are rejected. Additional
advertised contract versions are allowed. The current client selects request v1,
compiled system v1, result v2, and field array v1. It never silently downgrades a
negotiated result to v1. Historical result decoding remains available separately.

## Commands and capability checks

A solve command is one JSON line:

```json
{"operation":"solve","request":"/local/request.json","protocol_version":1,"result_schema_version":2}
```

A retained-field command selects its binary-array representation:

```json
{"operation":"bem_field","request":"/local/field.json","protocol_version":1,"field_array_schema_version":1}
```

The request file must remain available until the terminal event. Its mesh paths
and compiled payload follow the [compiled-system contract](BEAT%20Compiled%20System%20Contract.md).
Field binary filenames resolve against the field request's directory. All paths
are local to the worker; this protocol does not provide remote asset transfer.

Before submission the client checks the selected contract versions, operation,
precision, solve kind, requested backend, and marker cancellation support when
used. Interior FEM uses CPU; BEM backend selection applies to exterior and coupled
solves. Complex precision aliases remain accepted for interior/coupled requests.
An unavailable requested backend produces an error with the worker's reason;
negotiation does not choose a replacement backend. Application default-backend
selection remains a separate policy.

Feature-specific numerical options, boundary/component support, mesh validity,
and formulation restrictions remain validated by the solve plan and numerical
implementation. The broad solve-kind capability is not a claim that every
physical graph is supported.

## Events, cancellation, and process reuse

One job occupies a worker at a time. After `ready`, each accepted command produces
zero or more `status`, `result`, or `field_result` events and exactly one terminal
event: `completed`, `cancelled`, or `failed`. Frequency results preserve their
complex values and requested excitation ordering. A result event must use the
selected system result version; a field result must contain `values_binary`.

Julia reports command and job exceptions as `failed`, with human-readable `error`
and code `worker_request_failed`. The code identifies the request failure category;
clients must not parse the exception prose as a stable error taxonomy. A failed
command consumes one submission and leaves the worker able to accept another.
Process exit before a terminal event is a transport error. Text logging on stdout
is surfaced as status; stderr is collected for diagnostics.

Cancellation is cooperative: the client creates the solve's `cancel_path` marker,
and Julia stops at its existing cancellation checkpoints. Already emitted results
remain valid partial results. There is no fixed cancellation latency guarantee.
`bem_field` does not advertise cooperative cancellation. Forced process termination
invalidates its caches and announcement.

After a terminal event, the process can be reused. Closing an iterator before its
terminal event discards that process, preventing unread results from becoming the
next job's events. Consumers must drain or close a submission iterator. Restarted
workers always announce and negotiate again. Invalid startup announcements are
discarded; a capability mismatch for an individual request leaves a compatible
running process available for other jobs. `worker_info` returns a copy of the
current process's announcement and resets on termination.

## Field arrays and cache lifetime

Field-array v1 uses file descriptors with `file`, byte `offset`, `nbytes`, `dtype`,
`shape`, `order: C`, and `byte_order: little`. Complex values have adjacent real
and imaginary components in `complex64` or `complex128`. Mesh triangle indices
are zero-based. Existing binary readers validate bounds, sizes, and layout.
System result v2 uses the corresponding typed base64 representation described in
the compiled-system contract.

The field geometry cache belongs to one worker process. It holds at most two
entries with least-recently-used eviction, and is cleared for solves, failed
worker commands, and process exit. Keys combine the client domain key, precision,
backend, symmetry, and quadrature order. A domain key must identify immutable
geometry; change it when coordinates or topology change. Clients continue sending
complete geometry so eviction or process restart can rebuild the cache. This is
an optimization, not a durable engine object handle.

## Integration boundary

`beat_worker.py` remains a model-independent, standard-library transport. It stores
the announcement and offers validation hooks. Boundary Lab's runtime adapter uses
the engine-owned `beat_contract.worker` negotiation code in those hooks.

Bare legacy announcements remain supported only for the separate source/Deploy
reference transports. A physical compiled-system or retained-field request cannot
use that compatibility path, including with a custom worker script. The bundled
physical worker requires its versioned announcement even during warm-up.
The direct one-shot debugging path is not a persistent worker session and retains
its existing compiled-system validation without handshake negotiation.
