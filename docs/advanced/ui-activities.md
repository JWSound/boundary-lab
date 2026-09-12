# Shared UI activity indicator

The main window owns `activities`, an `ActivityController`. Its status row
displays the most recently started active task beside a small animated ring.
The ring appears after 200 ms of continuous activity. Its space remains reserved
when idle, and activity messages temporarily cover the normal status label
without overwriting completion or error messages.

For synchronous GUI-thread work, use a scope:

```python
with window.activities.start("Inspecting meshes...") as activity:
    inventory = inspect_meshes()
    activity.update("Preparing preview...")
    prepare_preview(inventory)
window.show_status("Preview ready")
```

The handle finishes even when the scope raises an exception or returns early.
End the scope before opening a modal dialog for user interaction.

For background work, retain a handle in the GUI-side operation controller:

```python
self.activity = window.activities.start("Loading meshes...")

# In GUI-thread slots receiving worker signals:
self.activity.update(message)
self.activity.finish()  # On success, failure, or cancellation.
```

`finish()` is idempotent. Each handle owns only its own activity; completing one
task cannot clear another. Updating an older task does not move it ahead of a
newer task. When the newer task finishes, the older task's latest message returns.
Use queued signals to GUI-thread slots; do not call the controller or handles
directly from a worker thread. `clear()` resets presentation on window shutdown;
it does not cancel workers.

Geometry generation and solving are connected through their existing
`state_changed` signals. Running and cancelling states retain their handles;
completed, failed, and cancelled states release them. Project reading, generated
geometry restoration, preview preparation, and System inventory inspection use
`PreparationController` jobs. Waiting for user preferences or interacting with
the System dialog does not count as a busy task.

The indicator uses Qt painting and the current palette, so it needs no image
asset or separate light/dark icon files. `BusyIndicator` and `ActivityStatusBar`
can also be reused with another `ActivityController`.

The indicator itself does not move work off the GUI thread. For new expensive
operations, submit a snapshot-based job through `window.preparations`:

```python
window.preparations.submit(
    "my-operation", "Preparing...",
    lambda: prepare(snapshot),  # No widgets or mutable live project state.
    apply_result,              # GUI-thread callback.
    show_error,                # GUI-thread callback.
)
```

Jobs run serially. Submitting the same key supersedes its previous request.
The GUI callback must also check that its project/snapshot is still current.
Cancelled or superseded jobs cannot publish results or errors. Stop skips queued
work and discards results from running work; an already-running parse is allowed
to finish. The indicator and mutation controls remain busy until it does.
Closing the window waits for running jobs to finish safely.

Widget construction, scene painting, plot clearing, and legacy source-model
migration still run on the GUI thread. An entirely synchronous scope may finish
before its delayed indicator can appear. Do not add `processEvents()` calls to
force animation inside application workflows.

See [mesh preparation and caching](mesh-preparation-performance.md) for cache
invalidation and measurement details.
