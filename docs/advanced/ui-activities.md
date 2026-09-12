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
completed, failed, and cancelled states release them. Project reading/loading
and System preparation use scopes. Waiting for user preferences or interacting
with the System dialog does not count as a busy task.

The indicator uses Qt painting and the current palette, so it needs no image
asset or separate light/dark icon files. `BusyIndicator` and `ActivityStatusBar`
can also be reused with another `ActivityController`.

This feature does not move work off the GUI thread. Synchronous mesh processing
still prevents timers and painting from running, and an entirely synchronous
scope may finish before its delayed indicator can appear. Do not add
`processEvents()` calls to force animation: migrate expensive preparation into
workers and publish results through GUI-thread slots instead.
