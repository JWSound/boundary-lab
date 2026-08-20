# Compression driver interior-FEM fixture

This fixture contains independent front- and rear-chamber tetrahedral meshes.
One electrodynamic component couples both diaphragm surfaces; rigid translation
is projected onto each chamber's computed outward normal. The front tube ends
in a plane-wave tube termination, so the project intentionally has no
unbounded exterior region and no interface.

From the repository root:

```bash
blab project validate examples/compression_driver/compression_driver.blab.json --json
blab project solve examples/compression_driver/compression_driver.blab.json \
  --request examples/compression_driver/validation-request.json \
  --backend beat_cpu --output runs/compression-driver-validation
```

The validation request solves one 1 kHz point and retains the concatenated FEM
nodal pressure for both regions.
