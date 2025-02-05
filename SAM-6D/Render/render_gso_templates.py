import blenderproc as bproc

import os
import cv2
import numpy as np
import trimesh
import time
import sys

# set relative path of Data folder
render_dir = os.path.dirname(os.path.abspath(__file__))
gso_path = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-GSO/google_scanned_objects')
gso_norm_path = os.path.join(gso_path, 'models_normalized')
output_dir = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-GSO/templates')
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

def format_duration(seconds):
    """
    Convert a duration in seconds to a string of the form:
      Dd HH:MM:SS (if days > 0)
      HH:MM:SS    (otherwise)
    """
    days = int(seconds // 86400)        # number of whole days
    seconds = int(seconds % 86400)      # remainder after days
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def custom_progress_bar(current, total, start_time):
    """
    Display a custom progress bar with ETA using `print`.
    Shows days when the duration is longer than 24 hours.

    Args:
        current (int): Current progress (e.g., 5 out of 100).
        total (int): Total iterations (e.g., 100).
        start_time (float): The start time of the process (time.time()).
    """
    # Calculate progress percentage
    progress = current / total
    bar_length = 40  # Length of the progress bar
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    # Calculate elapsed time and ETA
    elapsed_time = time.time() - start_time
    # Avoid division by zero if current == 0
    if current > 0:
        eta = (elapsed_time / current) * (total - current)
    else:
        eta = 0

    # Format times (supports days)
    elapsed_formatted = format_duration(elapsed_time)
    eta_formatted = format_duration(eta)

    # Print progress bar
    print(f"\r[{bar}] {current}/{total} - Elapsed: {elapsed_formatted} - ETA: {eta_formatted}")

    # Print a new line at the end
    if current == total - 1:
        print()      


def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)

def pair_opposite_coords(coords, tol=1e-6):
    """
    Given an Nx3 NumPy array 'coords' of 3D points, finds
    which points are approximate negatives of each other
    and arranges them in pairs next to each other.

    Returns:
      permuted_coords: Nx3 array with each opposite pair adjacent
      permutation:     list of indices used to permute the original array
                       so that pairs (i, j) appear consecutively as [i, j, ...].
    """
    n = len(coords)
    used = set()
    pairs = []

    for i in range(n):
        if i in used:
            continue
        vi = coords[i]
        # Find j such that coords[j] ~ -vi
        found_pair = False
        for j in range(i+1, n):
            if j in used:
                continue
            vj = coords[j]
            if np.allclose(vj, -vi, atol=tol):
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                found_pair = True
                break
        # In case there's a coordinate that doesn't have a negative match
        # (unlikely in a perfectly antipodal set, but just as a fallback)
        if not found_pair:
            pairs.append((i, i))
            used.add(i)

    # Build the permutation list so that each pair is [i, j] in sequence
    permutation = []
    for (i, j) in pairs:
        permutation.append(i)
        if j != i:
            permutation.append(j)

    permuted_coords = coords[permutation]
    return permuted_coords, permutation

def get_cam_locs(cnos_cam_fpath):
    """
    Loads camera poses, adjusts their alignment, and scales them to a target format.

    Args:
        cnos_cam_fpath (str): Path to the file with camera poses.

    Returns:
        np.ndarray: Transformed and aligned camera positions.
    """
    # Load camera poses
    cams = np.load(cnos_cam_fpath)
    
    # Adjust rotation and scale translation
    cams[:, :3, 1:3] = -cams[:, :3, 1:3]
    cams[:, :3, -1] *= 0.002  # Scale translation
    
    # Extract camera positions
    pos = cams[:, :3, -1]
    
    # permute so that opposing poses come sequentially
    pos, _ = pair_opposite_coords(pos)
    
    # Align directions
    dir_orig = np.array([0, 0, -1]) / np.linalg.norm([0, 0, -1])
    dir_tgt = np.array([-1, -1, -1]) / np.linalg.norm([-1, -1, -1])
    axis = np.cross(dir_orig, dir_tgt)
    axis_mag = np.linalg.norm(axis)
    dot = np.dot(dir_orig, dir_tgt)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + skew + skew @ skew * ((1 - dot) / (axis_mag ** 2))
    
    # Apply rotation and normalize
    aligned_pos = pos @ rot.T
    aligned_pos /= abs(aligned_pos[0, 0])
    
    return aligned_pos

bproc.init()
location = np.array([[1.000000,1.000000,1.000000],
                    [-1.000000,-1.000000,-1.000000],
                    [1.224745,-1.224745,0.000000],
                    [-1.224745,1.224745,0.000000],
                    [0.707107,0.707107,-1.414214],
                    [-0.707107,-0.707107,1.414214]
                    ])

# count total objects
total_objects = 0
for model_name in os.listdir(gso_norm_path):
    model_fpath = os.path.join(gso_norm_path, model_name)
    if not os.path.isdir(model_fpath) or '.' in model_name:
        continue
    total_objects += 1
start_time = time.time()
cur_iter = -1

for model_idx, model_name in enumerate(sorted(os.listdir(gso_norm_path))):
    if not os.path.isdir(os.path.join(gso_norm_path, model_name)) or '.' in model_name:
        continue
    cur_iter += 1
    if cur_iter < 0: continue
    custom_progress_bar(cur_iter, total_objects, start_time)
    print('---------------------------'+str(model_name)+'-------------------------------------')
    
    save_fpath = os.path.join(output_dir, model_name)
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    obj_fpath = os.path.join(gso_norm_path, model_name, 'meshes', 'model.obj')
    if not os.path.exists(obj_fpath):
        continue

    scale = get_norm_info(obj_fpath)

    for idx, loc in enumerate(location):
        if idx <=1: continue
        print('---------------------------view:'+str(idx)+'-------------------------------------')
        bproc.clean_up()

        obj = bproc.loader.load_obj(obj_fpath)[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", model_idx)

        # set light
        light_energy = 1000
        light_scale = 3
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location([light_scale*v for v in loc])
        light1.set_energy(light_energy)

        # add cam and point to object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() - loc)
        cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

        bproc.renderer.set_max_amount_of_samples(50)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())

        # save rgb images
        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

        # save masks
        mask_0 = data["nocs"][0][..., -1]
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    
        # save nocs
        xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
        np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))