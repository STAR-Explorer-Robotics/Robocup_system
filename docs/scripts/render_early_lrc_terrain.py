#!/usr/bin/env python3
"""Render the complete early LRC / ICRA 2024 QRC terrain from Unitree MJCF.

Usage: python docs/scripts/render_early_lrc_terrain.py /path/to/qrc_map_sloped.xml
Requires MuJoCo, NumPy and Pillow; the source XML's meshes must be available.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


SOURCE_URL = "https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/a2/qrc_map_sloped.xml"


def prepare(source, width, height):
    root = ET.parse(source).getroot()
    # The upstream include adds an A2 robot, separate from the complete terrain.
    for include in root.findall("include"):
        if include.get("file") != "a2.xml":
            raise ValueError(f"Unexpected include: {include.attrib}")
        root.remove(include)
    compiler = root.find("compiler")
    meshdir = source.parent / compiler.get("meshdir", "")
    compiler.set("meshdir", str(meshdir.resolve()))
    assets = root.find("asset")
    mesh_hashes = {
        mesh.get("file"): hashlib.sha256((meshdir / mesh.get("file")).read_bytes()).hexdigest()
        for mesh in assets.findall("mesh")
    }
    visual = root.find("visual")
    visual.find("global").attrib.update(offwidth=str(width), offheight=str(height), fovy="38")
    visual.find("map").attrib.update(znear="0.01", zfar="100", shadowscale="1.5")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.35 0.35 0.35", specular="0.05 0.05 0.05")
    ET.SubElement(assets, "texture", name="preview_sky", type="skybox", builtin="flat", rgb1="1 1 1", rgb2="1 1 1", width="256", height="1536")
    for texture in assets.findall("texture"):
        if texture.get("name") == "tex_ground":
            texture.attrib.update(rgb1="0.94 0.96 0.98", rgb2="0.94 0.96 0.98")
    for material in assets.findall("material"):
        if material.get("name") == "mat_ground":
            material.attrib.update(specular="0", reflectance="0")
        elif material.get("name") == "mat_1":
            material.attrib.update(rgba="0.58 0.64 0.69 1", specular="0.12", shininess="0.15")
    world = root.find("worldbody")
    for light in world.findall("light"):
        world.remove(light)
    ET.SubElement(world, "light", name="preview_key", directional="true", pos="-3 4 12", dir="0.5 -0.65 -1", diffuse="0.6 0.6 0.6", specular="0.08 0.08 0.08")
    ET.SubElement(world, "light", name="preview_fill", directional="true", pos="4 -3 10", dir="-0.5 0.3 -1", diffuse="0.16 0.16 0.16", specular="0 0 0", castshadow="false")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected_geoms = len(root.findall(".//geom"))
    assert model.ngeom == expected_geoms
    return model, data, mesh_hashes


def terrain_pixels(segmentation, ground_id):
    mask = (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (segmentation[:, :, 0] != ground_id)
    return np.nonzero(mask)


def fit_camera(model, data, camera, ground_id):
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        renderer.enable_segmentation_rendering()
        for _ in range(7):
            renderer.update_scene(data, camera=camera)
            ys, xs = terrain_pixels(renderer.render(), ground_id)
            if not len(xs):
                raise RuntimeError("No terrain visible")
            if xs.min() == 0 or xs.max() == 639 or ys.min() == 0 or ys.max() == 359:
                camera.distance *= 1.2
                continue
            up = np.asarray(renderer.scene.camera[0].up)
            right = np.cross(renderer.scene.camera[0].forward, up)
            scale = 2 * camera.distance * np.tan(np.radians(model.vis.global_.fovy / 2)) / 360
            camera.lookat[:] += right * ((xs.min() + xs.max()) / 2 - 320) * scale
            camera.lookat[:] += up * (180 - (ys.min() + ys.max()) / 2) * scale
            camera.distance *= max((xs.max() - xs.min() + 1) / (640 * 0.88), (ys.max() - ys.min() + 1) / (360 * 0.84))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "images" / "early_lrc_full_terrain.png")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--azimuth", type=float, default=100)
    parser.add_argument("--elevation", type=float, default=-60)
    args = parser.parse_args()
    source = args.source.resolve()
    model, data, mesh_hashes = prepare(source, args.width, args.height)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.2, 0, 0.3]
    camera.distance = 28
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    fit_camera(model, data, camera, ground)
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        renderer.update_scene(data, camera=camera)
        # Avoid shadow-map artifacts on the upstream scene's large ground plane.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        ys, xs = terrain_pixels(renderer.render(), ground)
    bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    assert bounds[0] > 0 and bounds[1] > 0 and bounds[2] < args.width - 1 and bounds[3] < args.height - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output, dpi=(300, 300))
    audit = {
        "source_url": SOURCE_URL,
        "source_xml_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_model": "ICRA2024_QRC_sloped",
        "mesh_sha256": mesh_hashes,
        "mujoco_version": mujoco.__version__,
        "terrain_body_count": model.nbody - 1,
        "terrain_geom_count": model.ngeom - 1,
        "presentation": "Complete terrain geometry and placement retained. A2 robot omitted; camera, lighting, materials and background adjusted, cast shadows disabled.",
        "camera": {"lookat": camera.lookat.tolist(), "distance": float(camera.distance), "azimuth": float(camera.azimuth), "elevation": float(camera.elevation), "fovy": float(model.vis.global_.fovy)},
        "image_size_px": [args.width, args.height],
        "terrain_bounds_px": bounds,
        "png_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved {args.output}: {model.nbody - 1} terrain bodies, {model.ngeom - 1} terrain geoms, bounds {bounds}")


if __name__ == "__main__":
    main()
