# Sample Assets

This directory contains example configuration files, mesh files, and scripts for testing and demonstrating Boundary Lab capabilities.

## Files

### Mesh Files (`.msh`)
- **waveguideexample.msh** - Example waveguide mesh for single-radiator testing
- **ath14symmetrysample.msh** - Symmetry sample mesh
- **athcomplexsample.msh** - Complex geometry sample
- **enclosureexample.msh** - Enclosure geometry example
- **multisource_example.msh** - Multi-source/radiator example mesh

### Configuration Files (`.cfg`)
- **ath.cfg** - Example ATH4 configuration
- **sampleathscript.cfg** - Sample configuration for ATH4 scripts
- **multisourceexample.cfg** - Multi-source solver configuration

### Scripts
- **sampleathscript/** - Sample ATH4 script directory with example workflow

## Usage

### Quick Start

Run a quick solve on a mesh:
```bash
blab clean waveguideexample.msh waveguideexample_clean.msh
blab solve waveguideexample_clean.msh --freq-min 200 --freq-max 20000 --freq-count 72
```

### Working with Configurations

Use configuration files for advanced multi-mesh or multi-radiator setups:
```bash
blab solve --config multisourceexample.cfg
```

## Notes

- These are provided as reference materials and starting points for your own simulations
- Mesh files follow Gmsh 2.2 format for compatibility
- All configurations can be modified to suit your specific requirements
- See the main README.md for detailed workflow documentation
