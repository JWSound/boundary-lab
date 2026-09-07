# Legacy server retirement

The source-model runtime, Bempp backend, and legacy HTTP solve server are retired.
Use a `.blab.json` physical-system project with BEAT Engine:

```bash
blab project validate speaker.blab.json --json
blab project solve speaker.blab.json --backend beat_cpu --output runs/speaker
```

Open older projects in the GUI to migrate their source assignments to a physical
system. Review the migrated component/channel assignments before solving. Existing
result readers and postprocessing remain available. Remote physical-system solving
will require a new service built on the physical-system contract; the retired HTTP
protocol is not a supported execution path.
