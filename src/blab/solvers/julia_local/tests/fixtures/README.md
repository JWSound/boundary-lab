# Frozen coupled reference meshes

These two meshes are preserved verbatim from the application fixture paths and
Git revision recorded in `manifest.json`. SHA-256 checks run before numerical
references. Keep the repository `LICENSE` with standalone source distributions.
The local `.gitattributes` disables mesh line-ending conversion so these hashes
remain valid across Windows, Linux, and macOS checkouts.

`femvolume.msh` is the 842-vertex, 2925-tetrahedron volume reference, with Volume,
Radiator, and Interface physical groups. `exterior_conforming.msh` supplies the
matching BEM surface. Both use millimeter coordinates and are loaded with scale
0.001. They preserve the existing FEM matrices, interface checks, coupled solves,
transducer operators, and condensed-versus-monolithic comparisons.

The application keeps its original fixtures for compiler/UI tests. These copies
are deliberately frozen numerical references, not a second authoring location.
Changing an application mesh does not automatically update this baseline. Any
reference update must explain its numerical effect, retain its origin, update the
manifest deliberately, and pass the analytical and cross-formulation checks.
There is no automatic baseline-update command.
