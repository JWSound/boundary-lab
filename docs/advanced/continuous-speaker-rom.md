# Continuous speaker package data

Every fidelity level includes a continuous Level 1 pattern of radiated complex pressure for each cabinet excitation port.
Level 3 provides the interior response to terminal voltage and external
acoustic loading. Both representations cover the project frequency band.

```powershell
$project = 'examples/S218BP/S218BP.blab.json'
blab project validate $project --json
blab project export-speaker $project --output S218BP-pattern.blabsp
blab project export-speaker $project --fidelity coupled --output S218BP-coupled.blabsp
```

## Level 1 pattern superposition

The `continuous_complex_spherical_pattern` capability identifies a Level 1
continuous pattern. Read the NPZ payload named by `files.patterns.path` in the
manifest. It stores the retained frequencies, sphere directions, observation
radii, and complex pressure in Pa with axes (frequency, excitation, direction).
The descriptor records the evaluator, frequency band in Hz, direct solve count,
error tolerance, and results of the midpoint and off-centre checks.

The `delay_compensated_linear_v1` evaluator accepts frequencies within
`files.patterns.frequency_band_hz`. At a retained frequency, use the stored
pressure. For a frequency *f* between two retained frequencies *a* and *b*,
evaluate each excitation and direction using the positive-time convention:

```python
t = (f - a) / (b - a)
delay_s = radius_m / sound_speed_m_per_s
pressure = ((1 - t) * pressure_a * np.exp(2j * np.pi * (a - f) * delay_s)
            + t * pressure_b * np.exp(2j * np.pi * (b - f) * delay_s))
```

This gives pressure at the stored radius. Apply receiver propagation and the
complex source drives when summing cabinets.

## Level 3 interior coupling

For the `continuous_speaker_rom` capability, read the model named by
`files.coupled_model.path`. The descriptor records the interior frequency
band, retained rank, training frequencies, and validation results.
In Levels 2 and 3, the top-level `frequencies_hz` list indexes the fixed-source
traces. The `frequencies_hz` array inside the pattern NPZ independently indexes
the pressure samples used for continuous pattern evaluation.

At an in-band frequency, supply the total complex surface pressure in Pa and
the voltage at each excitation port in manifest order. The model returns the
outward normal pressure derivative in Pa/m, diaphragm velocity, and terminal
current. Each column is an independent load case. In an array calculation the
surface pressure comes from the shared exterior system.

The BEAT Engine Python reader in `scripts/continuous_speaker_reader.py` can
evaluate the drive contribution with prescribed zero surface pressure:

```python
import numpy as np
from continuous_speaker_reader import ContinuousSpeaker

speaker = ContinuousSpeaker("S218BP-coupled.blabsp")
p = np.zeros((len(speaker.points_m), 1), dtype=complex)
v = np.zeros((len(speaker.ports), 1), dtype=complex)
v[0, 0] = 1.0
flux, velocity, current = speaker.evaluate(107.3, p, v)
```
