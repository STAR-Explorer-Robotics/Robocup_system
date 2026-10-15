# Localization

Two existing estimator foundations are preserved as separate backends:

| Directory | Role |
| --- | --- |
| [superodom](superodom/README.md) | SuperOdom/SuperLoc-based estimation, prior-map localization and JT128 adaptation |
| [lightning_lm](lightning_lm/README.md) | Lightning-LM SLAM/localization and the current online terrain-input interfaces |

Select a backend explicitly. Body reference, LiDAR reference, map frame and point timestamps must be aligned with downstream consumers. These forks are not presented as new estimation algorithms.

Each backend keeps its internal source layout, configuration and message/service definitions. Machine-specific launch scripts, precompiled binaries, maps and upstream demonstration media are excluded.

Build/dependency and licensing notes are in the backend READMEs and the [shared license inventory](../LICENSES.md).

