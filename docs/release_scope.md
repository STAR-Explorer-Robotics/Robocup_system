# Release scope

## Edition 0.1 — source organization

This edition is a curated source snapshot prepared on 2026-10-15. Its purpose is to make the modules, algorithms and interfaces understandable and reviewable without distributing the entire competition working directory.

Acceptance for this edition covers:

- selected module source and necessary local definitions are present;
- generated files, private deployment configuration and manuscript material are excluded;
- each module has a concise README and its upstream/source basis is recorded;
- copied-source changes are listed in the manifest;
- local documentation links and Python/configuration syntax are checked.

Building all modules, running an integrated navigation system, and executing on a robot are outside this edition's validation claim. Previous checks on the development tree do not certify this reorganized layout.

## Included and excluded

Included: estimator forks, geometry/terrain processing, graph planning, path tracking, the robot command adapter, task coordination, sensing runtime, representative parameters, message/service definitions, and existing small C++ component tests. The subsequent [terrain asset release](../assets/terrains/README.md) adds four MuJoCo terrains and their preview image.

Excluded: competition launch orchestration, remote-host/IP setup, calibration sessions and their results, recorded runs, other datasets/maps, experimental batch scripts, prebuilt tools, dependency archives, large upstream demonstration media, nested Git histories and agent instructions/state.

Inference model files already supplied with the sensing runtime are retained for continuity. Their file hashes are recorded; their presence is not an additional accuracy or provenance claim.

## Provenance and chronology

The baseline is the current local working tree, including uncommitted changes. Per-module Git HEAD values are provenance anchors; they do not imply that all exported files equal that commit.

The competition-era cutoff is the end of RoboCup 2026, 2026-07-06. The manifest does not infer feature introduction dates from file modification times. Features with insufficient dated evidence remain timing unconfirmed.

The manuscript, reviewer comments, partner-team architecture and full simulation/dataset resources are maintained separately.

## Before a public release

Resolve the outstanding module license declarations listed in LICENSES.md, select the public repository location, and publish an explicit version. The resulting README should describe precisely the provided modules and validation scope.

