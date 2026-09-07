# BEAT standalone candidate integration

The extracted engine checkout is `E:\Code\BEAT_Engine`. It retains 105 relevant
historical commits from Boundary Lab at `01988025adeabec50a692891deb35ba7fec29aa4`.
Its own `docs/EXTRACTION.md` records the path mapping and original-to-filtered
commit map. The initial Python package candidate is `beat-engine==0.1.0rc1`.

## Use the local candidate

```powershell
python -m pip install -e E:\Code\BEAT_Engine
$env:BLAB_BEAT_ENGINE_DISTRIBUTION = 'external'
python -m beat_engine doctor --backend cpu --threads 2
python -m blab.cli project validate examples/Simple_Sealed/simple_sealed.blab.json --backend beat_cpu --json
```

Launch Boundary Lab from the same shell to use the candidate in the GUI. The
environment selection is read at process startup, so restart the application
after changing it. Python and Julia need to use the same environment/configuration
as ordinary Boundary Lab runs; installing the Python package does not install Julia.

The `engine` extra pins the candidate version. Until a release is published, first
install the local checkout or the wheel built in `E:\Code\BEAT_Engine\dist`.
The candidate is not being published to PyPI by this work.

External mode uses BEAT's public client, contract validators, and asset-path API.
Boundary Lab retains project compilation, backend selection policy, ROCm SDK
discovery, GUI/CLI integration, and result models. Missing or mismatched packages
produce an error rather than silently falling back to the bundled engine.

For the pre-release qualification period, the default remains `bundled` and its
files remain available. Return to it with:

```powershell
$env:BLAB_BEAT_ENGINE_DISTRIBUTION = 'bundled'
```

## Qualification and publication sequence

1. Run BEAT's independent Python tests, Julia contract/CPU suite, and the complete
   464-check numerical reference gate in the standalone checkout.
2. Run Boundary Lab's tests with external mode selected. Validate then solve the
   same headless CPU baseline; compare all complex arrays and verify the reported
   engine repository/runtime paths. Check the installed wheel independently of
   the editable checkout.
3. Create an empty GitHub repository for BEAT, push the preserved `main` history,
   and run its Windows/Linux/macOS CPU CI. GitHub creation is not available through
   the current connector and no authenticated GitHub CLI is installed locally.
4. After that exact commit passes CI, tag `v0.1.0rc1` and use the repository's
   release-candidate workflow to attach the wheel and source distribution. The
   workflow verifies that the tag matches the package version and CI passed for
   the tagged commit. No publication or tag is implied by a local package build.
5. Pin Boundary Lab to the published immutable release artifact, make external
   mode the normal dependency path, and remove the duplicated bundled numerical
   assets after final integration qualification. Retain any still-needed source
   reference harness and separate accelerator qualification.

The editable candidate and built wheel are development artifacts, not a substitute
for a published release dependency. No remote repository, release, or production
dependency cutover has been claimed before those publication steps finish.
