# Development and releases

Use scoped branches from `main` and pull requests targeting `main`. Keep main
usable; a merge is not a user release. There is no permanent dev branch. Only
create maintenance branches when an older supported version needs a fix, and
bring applicable fixes back into main. Prefer squash merges for scoped PRs;
use a merge commit when preserving existing branch history during migrations.

The three repositories release independently. BEAT owns numerical implementation
and worker contracts. Boundary Lab owns design and speaker-package export.
Deploy owns deployment and its installer. Applications pin a published BEAT wheel
and SHA-256. To change both layers, link PRs, test against a uniquely versioned
engine candidate, release BEAT, then merge the application dependency update.
Never replace an existing version with different package contents.

Use patch versions for compatible fixes, minor versions for features, and document
any breaking change explicitly while versions are 0.x. Once the public API is 1.x,
use major versions for breaking changes. File-format and worker schema versions
are separate contracts; preserve old supported fixtures and test round trips.

## Release procedure

1. Open a release PR with matching version metadata, dependency pins and
   `docs/releases/VERSION.md`. Engine versions also live in its package
   `__init__.py`; Deploy versions also live in both npm manifest/lock files.
2. Merge after required checks succeed. Wait for successful **push CI on main**
   for that exact merge commit, then create and push tag `vVERSION`.
3. Dispatch the Release candidate workflow with that existing tag. It validates
   metadata and main CI, builds and checks the package, and creates a draft release
   with artifacts and SHA256SUMS.txt. Existing releases are never overwritten.
4. Download and test the final artifacts. Complete relevant CPU/GPU and application
   integration qualification. Record exact artifact hashes and outcomes in release
   notes. Publish the draft only after qualification; mark `rcN` releases as
   prereleases and do not make them latest stable. A failed candidate receives a
   new version/tag after fixes, never a rewritten published tag.
5. Enable GitHub release immutability; complete all assets before publication.
   Application dependency adoption happens in separate PRs and on its own schedule.

Release signing credentials belong in the GitHub `release` environment. Untrusted
PRs run on hosted runners without release secrets. GPU qualification stays a
maintainer-triggered workflow on trusted hardware. Do not run arbitrary fork code
on persistent self-hosted machines.

Main is protected with required CI and PR review for contributors. The owner can
merge their own PR after checks without a second maintainer; this exception is
visible in GitHub history. Do not bypass failing checks. Resolve PR conversations
and delete completed topic branches.

## Stable installations and the dev retirement

Stable Windows installs select the latest published stable release, not main.
Existing installers from 0.4.3 or older must be replaced with the `.bat` asset from
0.4.4 or newer **before** updating: old copies still pull main. A source ZIP has
no Git metadata; download a new stable release archive to update that installation.
To develop, clone the repository, create a scoped branch, and install `.[gui,dev]`
in a virtual environment. Do not use the stable updater to manage a feature branch.

The 0.4.4 maintenance release contains the updater migration only. Development
changes are reconciled afterward and do not enter stable until a later release.
