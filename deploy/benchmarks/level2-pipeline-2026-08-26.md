# Deploy Level 2 movement pipeline benchmark

Date: 2026-08-26

Hardware:

- AMD Ryzen 7 5700X, 8 cores / 16 threads
- NVIDIA GeForce RTX 2080 Ti, 11 GB, driver 596.49
- S218BP LOD package, two grounded cabinets, BEAT CUDA, rigid half-space ground

Run with:

```powershell
npm run build
npm run benchmark:level2
```

The Electron benchmark performs a cold 54 x 54 solve, moves one cabinet by
0.1 m for a warm 54 x 54 solve, then changes the audience plane to 200 x 200.
Each case reports interaction-to-texture latency and timestamps the
Julia/Python/Electron/renderer boundaries.

## Results

| Stage | Cold 54 x 54 | Warm move 54 x 54 | Warm plane 200 x 200 |
| --- | ---: | ---: | ---: |
| Interaction to texture | 38.472 s | 2.064 s | 2.416 s |
| Renderer IPC round trip | 38.100 s | 1.712 s | 1.904 s |
| Python request preparation | 1.097 s | 1.164 s | 1.124 s |
| Python waiting on Julia result | 35.985 s | 0.506 s | 0.668 s |
| Julia measured total before emit | 19.909 s | 0.475 s | 0.548 s |
| Julia operator assembly | 11.852 s | 0.163 s | 0.155 s |
| Julia solve | 2.797 s | 0.017 s | 0.017 s |
| Julia field evaluation | 1.620 s | 0.002 s | 0.028 s |
| Julia request JSON size | 1.06 MB | 1.06 MB | 3.43 MB |
| Julia result JSON size | 109 KB | 109 KB | 1.53 MB |
| Renderer field-frame parsing | 1.5 ms | 1.8 ms | 45.0 ms |
| Renderer heatmap rasterization | 3.7 ms | 3.1 ms | 57.3 ms |

The interaction measurement includes the 300 ms live-solve debounce. The cold
gap between Python's Julia wait and Julia's measured solve is primarily Julia
worker startup and compilation. The warm movement is dominated by Python
staging, not CUDA assembly or the renderer.

At 200 x 200, JSON decoding/encoding and renderer preparation become visible:
Julia-to-Python decode was 21.4 ms, Python result encoding 49.0 ms, Electron
decode 12.6 ms, field-frame construction 45.0 ms, and heatmap rasterization
57.3 ms. The renderer receives 160,000 numeric values although it only uses
the SPL and sample-index arrays; the complex field-pressure arrays account for
half of those values.

## Python preparation profile

A grounded two-cabinet staging profile took 1.51 s under `cProfile`:

- reflected ground-image face-pair traversal: 0.75 s;
- exact inter-cabinet minimum distance: 0.10 s;
- BVH construction: 0.18 s (included in the traversal totals);
- request JSON encoding: 0.018 s;
- package ZIP reads: 0.015 s.

The remaining exclusive time is largely the Python loop that filters shared
vertices and assigns correction tiers to roughly 50,000 conservative reflected
pair candidates.

## Recommended order of work

1. Cache package validation, extracted fixed-source arrays, topology, and base
   BVHs in the persistent Python worker. Refit or transform cached BVHs rather
   than rebuilding them for every movement.
2. Split plane-only updates from geometry solves. A resolution or plane-pose
   edit should reuse the current boundary pressure instead of rebuilding and
   solving all operators.
3. Cache Julia topology-dependent host/device data across speaker movements;
   update transformed vertices/normals and the near-pair list in place.
4. Stop returning complex field pressure to Deploy unless explicitly requested.
   Then move large observation/result arrays from JSON to typed binary buffers.
5. For 200 x 200 rendering, replace the full percentile sort with a selection
   or histogram pass, avoid allocating two Three.js `Color` objects per pixel,
   and update a persistent `DataTexture` buffer in place.
6. Build a Julia sysimage or eagerly warm the worker during application startup
   so the first interactive solve does not pay the compilation cost.
