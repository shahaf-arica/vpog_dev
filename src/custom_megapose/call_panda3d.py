from megapose.datasets.object_dataset import RigidObjectDataset, RigidObject
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.panda3d_renderer.types import Panda3dLightData
import logging
import numpy as np
from megapose.lib3d.transform import Transform
from PIL import Image
import os
import argparse
from src.utils.trimesh import get_obj_diameter
from bop_toolkit_lib import inout

#######################for debug ignore #########################################
# def mesh_to_depth(
#     cad_path: str,
#     T_cam_obj: np.ndarray,
#     K: np.ndarray,
#     resolution=(480, 640),
#     n_points: int = 200_000,
# ) -> np.ndarray:
#     """
#     Approximate depth map by projecting a sampled point cloud from the mesh.

#     Args:
#         cad_path: path to .obj
#         T_cam_obj: 4x4 transform that maps object coordinates -> camera coordinates
#                    (same convention as object_poses used for rendering)
#         K: 3x3 intrinsics matrix
#         resolution: (H, W)
#         n_points: number of random points sampled from the mesh surface

#     Returns:
#         depth: (H, W) float32 depth map in the same units as T_cam_obj translation.
#     """
#     import trimesh

#     H, W = resolution

#     # Load mesh
#     mesh = trimesh.load(cad_path, process=False)
#     if not isinstance(mesh, trimesh.Trimesh):
#         # e.g. a Scene; merge into one TriMesh
#         mesh = trimesh.util.concatenate(mesh.dump())

#     # Sample points on surface
#     # (you can also use mesh.vertices if you prefer, but sampling gives nicer coverage)
#     pts_obj, _ = trimesh.sample.sample_surface(mesh, n_points)  # (N, 3)

#     # Homogeneous coordinates in object frame
#     pts_obj_h = np.concatenate([pts_obj, np.ones((pts_obj.shape[0], 1))], axis=1).T  # (4, N)

#     # Transform to camera frame: X_cam = T_cam_obj * X_obj
#     pts_cam_h = T_cam_obj @ pts_obj_h  # (4, N)
#     pts_cam = pts_cam_h[:3]            # (3, N)

#     x = pts_cam[0]
#     y = pts_cam[1]
#     z = pts_cam[2]

#     # Keep only points in front of camera
#     valid = z > 0
#     x = x[valid]
#     y = y[valid]
#     z = z[valid]

#     if x.size == 0:
#         return np.zeros((H, W), dtype=np.float32)

#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]

#     # Project to pixels
#     u = (fx * x / z + cx)
#     v = (fy * y / z + cy)

#     # Round to nearest pixel
#     u = np.round(u).astype(np.int32)
#     v = np.round(v).astype(np.int32)

#     # Filter in-image
#     in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     u = u[in_img]
#     v = v[in_img]
#     z = z[in_img]

#     # Z-buffer (nearest depth per pixel)
#     depth = np.full((H, W), np.inf, dtype=np.float32)
#     flat_depth = depth.ravel()
#     idx = v * W + u
#     np.minimum.at(flat_depth, idx, z)
#     depth = flat_depth.reshape(H, W)

#     # Replace inf with 0 (like “no point”)
#     depth[~np.isfinite(depth)] = 0.0

#     return depth

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    parser.add_argument(
        "scale_translation", nargs="?", help="scale translation to meter"
    )
    parser.add_argument(
        "--point_light", nargs="?", type=float, help="point light level. if negative, use ambient light", default=-1.0
    )
    parser.add_argument(
        "--ambient_light", nargs="?", type=float, help="ambient light level", default=1.0
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus_devices)
    os.environ["EGL_VISIBLE_DEVICES"] = str(args.gpus_devices)

    label = 0
    is_shapeNet = "shapenet" in args.cad_path
    mesh_units = get_obj_diameter(args.cad_path) if not is_shapeNet else 1.0
    mesh_units = "m" if (mesh_units < 10 or is_shapeNet) else "mm"

    object = RigidObject(label=label, mesh_path=args.cad_path, mesh_units=mesh_units)
    rigid_object_dataset = RigidObjectDataset([object])

    # define camera
    disable_output = args.disable_output == "true" or args.disable_output == True
    renderer = Panda3dSceneRenderer(rigid_object_dataset, verbose=not disable_output)
    K = np.array([572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]).reshape(
        (3, 3)
    )
    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = -object_center
    TWC = Transform(camera_pose)
    camera_data = CameraData(K=K, TWC=TWC, resolution=(480, 640))

    point_light = args.point_light
    ambient_light = args.ambient_light

    if point_light < 0:
        # define light
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((ambient_light, ambient_light, ambient_light, 1)),
            ),
        ]
    else:
        def place_point_light_1(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            # light_np.set_pos(0.5, -0.8, 1.0)   # (x, y, z) in root_node coordinates
            light_np.set_pos(0.0, 0.0, 5.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        def place_point_light_2(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            light_np.set_pos(0.0, 5.0, 0.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        def place_point_light_3(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            light_np.set_pos(5.0, 0.0, 0.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        def place_point_light_4(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            light_np.set_pos(0.0, 0.0, -5.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        def place_point_light_5(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            light_np.set_pos(0.0, -5.0, 0.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        def place_point_light_6(root_node, light_np):
            # Put the light somewhere in front/above the object/camera
            light_np.set_pos(-5.0, 0.0, 0.0)   # (x, y, z) in root_node coordinates
            light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=(ambient_light, ambient_light, ambient_light, 1), # 0.7, 0.7, 0.7, 1
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_1,
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_2,
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_3,
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_4,
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_5,
            ),
            Panda3dLightData(
                light_type="point",
                color=(point_light, point_light, point_light, 1), # 0.8, 0.8, 0.8, 1
                positioning_function=place_point_light_6,
            ),
        ]

    # def place_point_light(root_node, light_np):
    #         # Put the light somewhere in front/above the object/camera
    #         light_np.set_pos(0.5, -0.8, 1.0)   # (x, y, z) in root_node coordinates
    #         light_np.look_at(0, 0, 0)          # aim toward origin (optional for point)

    # light_datas = [
    #     Panda3dLightData(
    #         light_type="ambient",
    #         color=(0.7, 0.7, 0.7, 1), # 0.7, 0.7, 0.7, 1
    #     ),
    #     Panda3dLightData(
    #         light_type="point",
    #         color=(0.8, 0.8, 0.8, 1), # 0.8, 0.8, 0.8, 1
    #         positioning_function=place_point_light,
    #     ),
    # ]

    
    # load object poses
    object_poses = np.load(args.obj_pose)
    if mesh_units == "m" or args.scale_translation == "true":
        object_poses[:, :3, 3] /= 1000.0

    for idx_view in range(len(object_poses)):
        TWO = Transform(object_poses[idx_view])
        object_datas = [ObjectData(label=label, TWO=TWO)]
        camera_data, object_datas = convert_scene_observation_to_panda3d(
            camera_data, object_datas
        )

        # render
        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=True,
            render_binary_mask=True,
            render_normals=True,
            copy_arrays=True,
            clear=True,
        )[0]

        # # ---- our depth from point cloud ----
        # pc_depth = mesh_to_depth(
        #     cad_path=args.cad_path,
        #     T_cam_obj=object_poses[idx_view],
        #     K=K,
        #     resolution=(480, 640),
        # )
        # pc_depth_nonzero = pc_depth[pc_depth > 0]
        # save rgba
        rgb = renderings.rgb
        mask = renderings.binary_mask * 255
        rgba = np.concatenate([rgb, mask[:, :, None]], axis=2)
        save_path = f"{args.output_dir}/{idx_view:06d}.png"
        Image.fromarray(np.uint8(rgba)).save(save_path)

        # save depth
        save_depth_path = f"{args.output_dir}/{idx_view:06d}_depth.png"
        if mesh_units == "m" or args.scale_translation == "true":
            renderings.depth *= 1000.0
        inout.save_depth(save_depth_path, renderings.depth)
        # renderings_depth_nonzero = renderings.depth[renderings.depth > 0]

    # count the number of rendering
    renderings = os.listdir(args.output_dir)
    renderings = [r for r in renderings if r.endswith(".png")]
    num_rendering = len(renderings)
    # print(f"Found {num_rendering}", args.output_dir)
    if len(object_poses) * 2 != num_rendering:
        print("Warning: the number of rendering is not equal to the number of poses")
