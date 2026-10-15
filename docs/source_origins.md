# Source origins

This edition was copied from the current local development checkout on 2026-10-15. The rows below identify source repositories and Git HEAD anchors where available. Working-tree modifications were included; a HEAD value alone is not the exact exported content.

| Module | Recorded source repository | HEAD anchor | Working tree modified |
| --- | --- | --- | --- |
| [localization/superodom](../localization/superodom/README.md) | [https://github.com/Ancientree-Bichy/SuperOdom](https://github.com/Ancientree-Bichy/SuperOdom) | 4287a6213de8 | No |
| [localization/lightning_lm](../localization/lightning_lm/README.md) | [https://github.com/LingJingHu/lightning-lm](https://github.com/LingJingHu/lightning-lm) | ced54f795d51 | Yes |
| [terrain_processing](../terrain_processing/README.md) | [https://github.com/Ancientree-Bichy/TRG-Planner-Python](https://github.com/Ancientree-Bichy/TRG-Planner-Python) | fed5ee1a26f0 | Yes |
| [terrain_planning](../terrain_planning/README.md) | [https://github.com/Ancientree-Bichy/TRG-planner](https://github.com/Ancientree-Bichy/TRG-planner) | c78bf8db12f1 | Yes |
| [path_following](../path_following/README.md) | [https://github.com/STAR-Explorer-Robotics/trg_path_follower](https://github.com/STAR-Explorer-Robotics/trg_path_follower) | b65627add17e | Yes |
| [robot_interface](../robot_interface/README.md) | [https://github.com/STAR-Explorer-Robotics/unitree_adapter](https://github.com/STAR-Explorer-Robotics/unitree_adapter) | 1b341ff25fc7 | Yes |
| [mission_control](../mission_control/README.md) | Current workspace source | No module Git HEAD in supplied tree | Not independently versioned |
| [perception](../perception/README.md) | Current workspace source | No module Git HEAD in supplied tree | Not independently versioned |

The ROS terrain bridge and robot-reference odometry bridge were copied from the workspace integration scripts and placed under terrain_processing/ros. They are included in the file-level manifest.

## Method attribution

- SuperOdom / SuperLoc: existing estimation and localization foundations; original references are cited by the SPARC paper and the upstream project.
- Lightning-LM: existing SLAM/localization foundation; this snapshot retains the working fork and its navigation output adaptations.
- TRG-planner: Dongkyu Lee, I Made Aswin Nahrendra, Minho Oh, Byeongho Yu and Hyun Myung, “TRG-planner: Traversal Risk Graph-Based Path Planning in Unstructured Environments for Safe and Efficient Navigation,” IEEE Robotics and Automation Letters, 2025. The imported source points to https://arxiv.org/abs/2501.01806.
- Unitree motion execution uses the official SDK2 API. The SDK and vendor locomotion controller are external components.

Original source copyright notices and available license files remain in the copied modules. See [LICENSES.md](../LICENSES.md) for declarations that need clarification.

## Snapshot records

- [source_snapshot.json](../metadata/source_snapshot.json): original relative paths, source hashes, exported hashes and module anchors.
- [release_edits.json](../metadata/release_edits.json): cleanup performed while preparing this layout.

Release edits cover documentation, packaging metadata, generic example paths and the terrain library lookup after relocation. The collection does not infer competition deployment dates from the copy date or file modification times.
