# Separate FEM and BEM mesh levels and global finenesses

CAD import has three FEM levels (Fine 6, Medium 4, Coarse 2 elements per wavelength at fmax) and three BEM levels (Fine 12, Medium 5, Coarse 0.3), and two global mesh finenesses: FEM for air-body volumes and faces, BEM for exterior surface faces, with the finer one on a face that is on both. FEM regions are best meshed uniformly at a size set by fmax. BEM meshes can be graded much more: they need small elements near HF sources and complex geometry (domes, phase plugs) but can be far coarser than a wavelength on the sides and rear of an enclosure, since the solver's order-4 quadrature samples 6 points per triangle. A forum post from an experienced user gave the BEM values as edge lengths (1.5 mm for domes and phase plugs, 4 mm near HF sources, 40–60 mm on the sides and rear). We turned them into elements per wavelength at 20 kHz (λ ≈ 17 mm), rounded so that no BEM level has the same number as an FEM level. This way, the levels still follow the project's maximum frequency. The BEM global defaults to Medium, because Fine over a whole cabinet means hundreds of thousands of elements. Designs saved before this change have no BEM global; they use their FEM value for it, so they mesh exactly as before.

## Considered Options

- Fixed BEM edge lengths (1.5 / 4 / 60 mm) that do not follow fmax: rejected, because a lower fmax would keep needlessly fine meshes.
- One global fineness for both methods: rejected, because it either over-refines the exterior or under-refines the interior.
- BEM Fine as the default BEM global: rejected because of the element count. The guidance is to refine only HF sources and complex geometry.
