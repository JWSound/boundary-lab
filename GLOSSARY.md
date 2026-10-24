# Boundary Lab

Loudspeaker acoustic simulation: a physical system of air regions, their
boundaries, and the devices that drive them, solved with BEM/FEM.

## Solving

**Warm-up**:
One-off solver setup before the first frequency result: worker start, compilation, geometry caches.
_Avoid_: Caching, initializing

**Frequency solve time**:
Wall-clock time from one frequency result to the next; the first gap is excluded because it contains pipeline fill.
_Avoid_: Frequency step, per-frequency time
