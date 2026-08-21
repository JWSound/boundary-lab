# macOS HornLab Integration

Boundary Lab runs Ath through Wine on macOS and Linux. Two optional HornLab
packages remove that dependency for the geometry families they support and add a
native Apple Silicon solver backend. Windows installs are unaffected.

## What Gets Installed

`python -m pip install -e ".[gui]"` installs, from pinned public GitHub
revisions:

| Package | Installed on | Purpose |
| --- | --- | --- |
| [`hornlab-waveguide-mesher`](https://github.com/m3gnus/hornlab-waveguide-mesher) | any non-Windows platform | Native OSSE/R-OSSE waveguide meshing without Wine or Ath |
| [`hornlab-metal-bem`](https://github.com/m3gnus/hornlab-metal-bem) | Apple Silicon macOS | `HornLab Metal BEM` solver backend |

No sibling HornLab checkout is required. Verify an installation with:

```bash
python -c "import hornlab_mesher; print('HornLab mesher ready')"
python -c "import hornlab_metal_bem; print('HornLab Metal ready')"
```

The Metal package ships a Swift source fallback, so a precompiled `.metallib` is
not required. Building the optional precompiled library needs a Metal-capable
Xcode toolchain.

## Waveguide Meshing

Ath geometry generation first tries the HornLab mesher, then falls back to
Ath/Wine. Both paths return the same `AthRunResult`: the mesher's raw mesh is
handed to Boundary Lab's shared cleaning and mirroring stage, so cleaned,
reduced, and expanded artifacts are produced exactly as they are for Ath.

Boundary Lab routes a config to Ath instead when:

- `Throat.Ext.Length` requests a throat extension tube. The pinned mesher builds
  OSSE and R-OSSE extensions, but Boundary Lab has no test pinning that geometry
  against Ath, and older mesher releases silently shrank the horn.
- `Source.Contours`, `LFSource.*`, or `Source.Velocity.*` define a multi-source
  model. The mesher builds only the single throat source.
- `Mesh.SubdomainSlices`/`Mesh.InterfaceOffset` appear on a free-standing model
  (`ABEC.SimType = 2`). Ath builds a two-subdomain model there; the mesher
  rejects that topology rather than silently building a different one.
- The mesher is not installed, or fails for any other reason.

If the mesher reports different quadrant coverage than `Mesh.Quadrants` implies,
the run is routed to Ath rather than mirrored against the wrong axes.

Each mesher-generated run writes `hornlab_mesher.json` next to the mesh,
recording the mesher version, formula, mode, units, quadrants, and element
counts, so a run's geometry can be reproduced later.

## HornLab Metal BEM Backend

The backend is registered as `hornlab_metal` (aliases: `hornlab`, `metal`,
`apple_metal`) and is listed only when `hornlab-metal-bem` is importable and
reports Apple Silicon support. Its package version is recorded in each run's
solve provenance.

Two behavioural notes:

- The Metal solver has no Burton-Miller operator. Its adapter maps that
  robustness request onto the solver's complex-wavenumber formulation.
- The solver's open-edge guard requires every open edge of a mirror-reduced mesh
  to lie on a symmetry plane. An open-mouth (bare) horn's mouth rim is a real
  free edge, so Boundary Lab clears `native_check_open_edges` for generated bare
  waveguides and keeps the strict check everywhere else.

## Known Limits

- The mesher is not a full Ath replacement; unsupported geometry still needs Wine
  and Ath.
- The Metal backend is Apple Silicon only. Intel Macs use the existing CPU
  backends.
