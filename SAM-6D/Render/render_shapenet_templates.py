#!/usr/bin/env python3

import blenderproc as bproc
import os
import sys
import cv2
import numpy as np
import trimesh
import time

# -------------------------------------------------------------------------
# 1) Parse --chunk_file argument and read (synset_id, source_id) pairs
# -------------------------------------------------------------------------
chunk_file = None
argv = sys.argv
for i, arg in enumerate(argv):
    if arg == "--chunk_file":
        chunk_file = argv[i + 1]
        break

if chunk_file is None:
    print("ERROR: No --chunk_file argument provided! Use: --chunk_file chunk_0.txt")
    sys.exit(1)

pairs_to_render = []
with open(chunk_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            synset_id, source_id = line.split(',')
            pairs_to_render.append((synset_id, source_id))

# -------------------------------------------------------------------------
# 2) Setup all other paths and helper functions
# -------------------------------------------------------------------------
render_dir = os.path.dirname(os.path.abspath(__file__))

# Adjust these paths to match your project’s structure:
shapenet_path = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2')
shapenet_orig_path = os.path.join(shapenet_path, 'models_orig')
output_dir = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/templates')
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

def custom_progress_bar(current, total, start_time):
    """
    Display a custom progress bar with ETA using `print`.
    """
    progress = current / total
    bar_length = 40
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    elapsed_time = time.time() - start_time
    eta = (elapsed_time / (current + 1)) * (total - current - 1) if current > 0 else 0

    elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta))

    print(f'\r[{bar}] {current}/{total} - Elapsed: {elapsed_formatted} - ETA: {eta_formatted}', end='')
    if current == total - 1:
        print()

def get_norm_info(mesh_path):
    """
    Loads the mesh with trimesh, samples points, finds bounding extremes,
    and returns the appropriate scale factor (1 / (2*radius)).
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    # Sample ~1024 points on the surface
    model_points = trimesh.sample.sample_surface(mesh, 1024)[0].astype(np.float32)
    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    return 1/(2*radius)

def get_cam_locs(cnos_cam_fpath):
    """
    Loads camera poses, adjusts their alignment, and scales them to a target format.
    Returns an (N,3) array of camera location vectors.
    """
    cams = np.load(cnos_cam_fpath)
    # Adjust rotation and scale translation
    cams[:, :3, 1:3] = -cams[:, :3, 1:3]
    cams[:, :3, -1] *= 0.002

    # Extract camera positions
    pos = cams[:, :3, -1]

    # Align directions
    dir_orig = np.array([0, 0, -1]) / np.linalg.norm([0, 0, -1])
    dir_tgt = np.array([-1, -1, -1]) / np.linalg.norm([-1, -1, -1])
    axis = np.cross(dir_orig, dir_tgt)
    axis_mag = np.linalg.norm(axis)
    dot = np.dot(dir_orig, dir_tgt)
    skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rot = np.eye(3) + skew + skew @ skew * ((1 - dot) / (axis_mag ** 2))

    aligned_pos = pos @ rot.T
    aligned_pos /= aligned_pos[0, 0]  # Scale by first position
    # Move last to top
    aligned_pos = np.vstack([aligned_pos[-1:], aligned_pos[:-1]])
    return aligned_pos

# -------------------------------------------------------------------------
# 3) Initialize BlenderProc & precompute camera positions
# -------------------------------------------------------------------------
bproc.init()
location = get_cam_locs(cnos_cam_fpath)

# For progress bar
start_time = time.time()
total_objects = len(pairs_to_render)

# -------------------------------------------------------------------------
# 4) Loop over all pairs in this chunk & render
# -------------------------------------------------------------------------
for idx, (synset_id, source_id) in enumerate(pairs_to_render):
    custom_progress_bar(idx, total_objects, start_time)

    # Paths
    save_synset_folder = os.path.join(output_dir, synset_id)
    os.makedirs(save_synset_folder, exist_ok=True)

    save_fpath = os.path.join(save_synset_folder, source_id)
    os.makedirs(save_fpath, exist_ok=True)

    cad_path = os.path.join(shapenet_orig_path, synset_id, source_id)
    obj_fpath = os.path.join(cad_path, 'models', 'model_normalized.obj')
    if not os.path.exists(obj_fpath):
        continue

    # Check if already rendered
    rgb42path = os.path.join(save_fpath, 'rgb_41.png')
    if os.path.exists(rgb42path):
        print(f"--already rendered-- {synset_id}/{source_id}")
        continue

    # Compute the scale factor for this object
    scale = get_norm_info(obj_fpath)

    # Clean up the scene from any prior iteration
    bproc.clean_up()

    # Load the object from ShapeNet
    obj = bproc.loader.load_shapenet(
        shapenet_orig_path, 
        synset_id, 
        source_id, 
        move_object_origin=False
    )
    obj.set_scale([scale, scale, scale])
    # Optional: set a category ID
    obj.set_cp("category_id", idx)

    # Create lights, one for each camera location (if desired)
    light_energy = 60
    light_scale = 3
    lights = []
    for loc_i, loc in enumerate(location):
        l = bproc.types.Light()
        l.set_type("POINT")
        l.set_location([light_scale * v for v in loc])
        l.set_energy(light_energy)
        lights.append(l)

    # Add one camera pose for each location
    for loc in location:
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            obj.get_location() - loc
        )
        cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

    # Minimal sample count to speed up rendering
    bproc.renderer.set_max_amount_of_samples(1)

    # Render the pipeline (standard)
    data = bproc.renderer.render()
    # Render NOCS
    data.update(bproc.renderer.render_nocs())

    # Save outputs for each camera
    for view_idx, loc in enumerate(location):
        # Save RGB (flip from BGR -> RGB if needed)
        color_bgr = data["colors"][view_idx]
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath, f'rgb_{view_idx}.png'), color_bgr)

        # Save mask (last channel of NOCS)
        mask_img = data["nocs"][view_idx][..., -1]
        cv2.imwrite(os.path.join(save_fpath, f'mask_{view_idx}.png'), mask_img * 255)

        # Save NOCS as .npy
        xyz_nocs = 2 * (data["nocs"][view_idx][..., :3] - 0.5)
        # Rotate 90 degrees to match your specific CAD orientation
        rot90 = np.array([
            [1,  0, 0],
            [0,  0, 1],
            [0, -1, 0]
        ])
        h, w = xyz_nocs.shape[:2]
        xyz_nocs_reshaped = xyz_nocs.reshape(-1, 3)
        xyz_nocs_rotated = (rot90 @ xyz_nocs_reshaped.T).T.reshape(h, w, 3)
        np.save(
            os.path.join(save_fpath, f'xyz_{view_idx}.npy'),
            xyz_nocs_rotated.astype(np.float16)
        )

print("\nDone rendering all pairs in this chunk.")