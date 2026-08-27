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
0.1 m for a warm 54 x 54 solve, changes the audience plane to 200 x 200, then
moves that plane by 0.1 m for a repeated 200 x 200 evaluation. Each case
reports interaction-to-texture latency and timestamps the
Julia/Python/Electron/renderer boundaries.

## Results

### Baseline

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

### After the first performance pass

The optimized run uses persistent package and ground-image-pair caches, retains
the current boundary solution for plane-only updates, omits unused complex
pressure, and avoids renderer sort/color allocation hot spots.

| Stage | Cold 54 x 54 | Warm move 54 x 54 | Warm plane 200 x 200 |
| --- | ---: | ---: | ---: |
| Interaction to texture | 38.394 s | 1.417 s | 0.792 s |
| Renderer IPC round trip | 37.964 s | 1.017 s | 0.457 s |
| Python request preparation | 0.613 s | 0.115 s | 0.062 s |
| Python waiting on Julia result | 24.446 s | 0.897 s | 0.356 s |
| Julia measured total before emit | 20.373 s | 0.499 s | 0.223 s |
| Julia operator assembly | 12.091 s | 0.163 s | 0 s |
| Julia solve | 2.875 s | 0.016 s | 0 s |
| Julia field evaluation | 1.592 s | 0.002 s | 0.143 s |
| Julia request JSON size | 1.06 MB | 1.06 MB | 2.56 MB |
| Julia result JSON size | 41 KB | 41 KB | 596 KB |
| Renderer field-frame parsing | 0.9 ms | 0.3 ms | 1.6 ms |
| Renderer heatmap rasterization | 1.4 ms | 0.3 ms | 2.0 ms |

The 200 x 200 plane case reports `field_only=1`; assembly and the boundary
solve are skipped. Relative to the baseline, warm cabinet interaction latency
improved by 31%, plane-only latency improved by 67%, and Python preparation for
a cabinet movement improved by 90%. Result transport is about 62% smaller for
54 x 54 and 61% smaller for 200 x 200 because Deploy no longer transfers an
unused complex-pressure copy.

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

## Implementation status

Implemented in this pass:

1. Persistent package validation/fixed-source caching and reusable canonical
   cabinet-to-ground near-pair lists, with cheap AABB rejection for cross-image
   pairs.
2. Plane-only field requests backed by a persistent Julia boundary state.
3. Optional complex-pressure transport, disabled by Deploy by default.
4. Typed-array percentile selection and allocation-free heatmap color mapping.
5. Eager CUDA Julia-worker startup when the desktop application opens.

The next useful performance tier is Julia topology/device-cache reuse across
cabinet movements, followed by binary observation/result buffers. A sysimage or
representative background JIT warmup remains the appropriate fix for cold-solve
latency; it does not affect the now-subsecond plane interaction path.

### GPU-resident observation evaluator

The next pass retained boundary pressure, Neumann data, and weighted field
sources on CUDA; generated compact plane coordinates on CUDA; and converted
complex pressure to SPL before copying results to the host. The grid kernel is
warmed during the initial solve so the first interactive plane edit does not
pay its compilation cost.

| Stage | Previous 200 x 200 | GPU first edit | GPU repeated edit |
| --- | ---: | ---: | ---: |
| Interaction to texture | 0.792 s | 0.592 s | 0.513 s |
| Renderer IPC round trip | 0.457 s | 0.204 s | 0.133 s |
| Python request preparation | 61.8 ms | 2.3 ms | 2.4 ms |
| Julia request JSON size | 2.56 MB | 808 bytes | 808 bytes |
| Julia measured total | 0.223 s | 0.090 s | 0.068 s |
| CUDA observation preparation | included in field | 33.3 ms | 38.8 ms |
| CUDA field/SPL evaluation | 143.2 ms | 28.2 ms | 28.4 ms |
| Result JSON size | 596 KB | 596 KB | 596 KB |

The first interactive 200 x 200 edit improved by 25%, and the repeated edit by
35%. The 300 ms live-solve debounce is included in those interaction totals.
The result side is now the dominant transport cost because JSON still carries
40,000 SPL values and 40,000 sample indices; binary SPL plus a compact validity
mask is the next observation-specific transport opportunity.

### Direct Burton-Miller system assembly

Deploy CUDA Level 2 now assembles the final Burton-Miller matrix and fixed-trace
right-hand side directly. It no longer retains separate dense S, D, D*, and H
operators or dense identity matrices. The operator-matrix path remains
selectable as a reference fallback.

For the same two-cabinet 54 x 54 movement case, warm direct assembly measured
0.080 s versus 0.163 s in the earlier operator-matrix run. The complete warm
cabinet interaction measured 1.11-1.18 s, while the in-Julia boundary update
measured 0.265 s. A 250 ms `nvidia-smi` sample during the cold run observed
1,484 MiB total device use from a 919 MiB desktop idle baseline, or about
565 MiB incremental. Compared with the earlier 1,768 MiB incremental
operator-matrix measurement, this is an observed reduction of about 68%.

These VRAM figures include the Electron renderer and CUDA allocator state and
are sampled rather than allocator-instrumented peaks, so they should be treated
as practical end-to-end measurements rather than exact matrix-allocation totals.
