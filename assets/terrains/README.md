# RB-TRG terrain assets

Four scanned terrain heightfields used in our [RB-TRG experiments](https://github.com/ZhiqiaoGuo/RB-TRG), representing **only part of the terrains at RoboCup/LRC**.

Each of [Scene 1](scene1/), [Scene 2](scene2/), [Scene 3](scene3/), and [Scene 4](scene4/) contains a standalone MuJoCo `scene.xml`, the original `heightfield.bin`, and `metadata.json` with dimensions and world bounds in meters. The heightfields retain the evaluation geometry at 0.02 m resolution; row indices increase along world +y. Source paths and SHA256 hashes are recorded in [manifest.json](manifest.json).

With MuJoCo installed, open a scene from the repository root:

```sh
python -m mujoco.viewer --mjcf=assets/terrains/scene1/scene.xml
```

The scenes contain terrain only. The [preview](../../docs/images/rb_trg_terrains.png) uses presentation lighting and cameras without changing the terrain geometry. Scene loading was checked with MuJoCo 3.3.6.

An additional [MuJoCo preview](../../docs/images/early_lrc_full_terrain.png) shows **the complete terrain layout from an early LRC edition (ICRA 2024 QRC)**, sourced from [Unitree's `qrc_map_sloped.xml`](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/a2/qrc_map_sloped.xml). The model and meshes are available upstream; this repository includes the screenshot, [rendering script](../../docs/scripts/render_early_lrc_terrain.py), and [source hashes and camera settings](../../docs/images/early_lrc_full_terrain.json).
