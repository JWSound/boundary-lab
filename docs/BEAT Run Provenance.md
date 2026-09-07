# BEAT engine and runtime provenance

Physical-system results now include `diagnostics.engine_provenance` with
`schema_version: 1`. This additive diagnostics field does not change the compiled
request, worker protocol, or result-array versions. Julia produces the record,
including in the direct one-shot solve path; the application does not infer the
actual execution backend from its requested backend preference.

## Engine identity

The worker's `ready.engine` and each result's provenance identify the engine using:

- `name` and `version`: the engine-owned interface/release identity.
- `repository_revision`: the containing checkout's Git HEAD, or null when Git
  metadata cannot be read. This is the Boundary Lab repository while BEAT remains
  bundled; after extraction it will identify the engine repository.
- `repository_dirty`: engine-scoped Git status, including untracked files, or null
  when unavailable. The scope is `julia_local` and its sibling `beat_contract`.
  Changes to application UI files alone do not mark the engine dirty.
- `source_files_sha256`: content hashes of root Julia entrypoints, Julia `src`
  files, and root Python/JSON files in `beat_contract`. Numerical fixtures and
  Julia tests are excluded from this content digest; they remain inside the Git
  dirty-status scope.
- `source_sha256`: SHA-256 of the sorted source-file records. Each record is UTF-8
  `relative/path`, a NUL separator, the lowercase hexadecimal file digest, and a
  newline. Relative paths use forward slashes and the `julia_local/` or
  `beat_contract/` prefix.

This content identity is available without Git and distinguishes modified source
trees that share a commit. It is a source fingerprint, not a signature or a claim
that an arbitrary sysimage was built from those sources. Keep the corresponding
checkout or release artifacts when reproducing results.

## Runtime and execution

`runtime` records Julia version and executable path; active Project and Manifest
paths and SHA-256 hashes; loaded sysimage path and hash; Julia thread count; BLAS
configuration and thread count; machine, OS kernel, and CPU identity. The manifest
lookup prefers `Manifest-v<major>.<minor>.toml`, then `Manifest.toml`. Missing files
are represented by null hashes. Environment files and the sysimage are identified
by content; they are not copied into the run archive.

Engine identity and the static runtime identity are captured once for the running
process. Restart the worker after changing its source, environment, or sysimage.
BLAS thread count is sampled again for results because the solver adjusts it for
the numerical workload.

`execution` records the backend and precision reported by numerical diagnostics,
the current accelerator device string (or CPU identity), and the submitted
numerical options. Failed device queries leave a null device and an explicit
`device_query_error`; the client never substitutes the requested device as if it
were observed. Detailed solver diagnostics continue to identify internal solver
choices and precision distinctions, such as an FP64 sparse factorization.

`meshes` records each input mesh ID, worker-local filename, and SHA-256 hash sampled
at solve entry. Headless runs also retain their existing compiled graph, transforms,
project snapshot, requested options, and application-side mesh hashes. Inputs must
remain unchanged during a solve; these hashes do not turn mutable files into an
atomic snapshot. General process environment variables are not recorded.

## Artifact propagation

- Frequency JSON metadata retains the full result diagnostics.
- Headless `manifest.json` stores distinct records in `engine_runs`, and the
  worker announcement in `worker` when one was captured. A failure/cancellation
  before the first result can retain the announcement without claiming an actual
  numerical execution. No worker announcement is synthesized for one-shot solves.
- `SolvedSystemBuilder` collects distinct records into `SolveProvenance.engine_runs`
  for GUI and headless package workflows. Snapshots copy provenance so later
  mutations cannot alter previous snapshots.
- Speaker-package manifests preserve `provenance.engine_runs`.

Distinct records are retained rather than replacing the first record if execution
details differ later in a run. Per-frequency diagnostics identify which record
belongs to each result. Older results and synthetic/reference backends without
engine provenance remain readable and have an empty collection; missing identity
is never presented as a verified revision or device.
