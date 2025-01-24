import blenderproc as bproc

import os
import cv2
import numpy as np
import trimesh
import time
import sys

# set relative path of Data folder
render_dir = os.path.dirname(os.path.abspath(__file__))
shapenet_path = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2')
shapenet_orig_path = os.path.join(shapenet_path, 'models_orig')
output_dir = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/templates')
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
location = get_cam_locs(cnos_cam_fpath)

# count total objects
total_objects = 0
for synset_id in os.listdir(shapenet_orig_path):
    synset_fpath = os.path.join(shapenet_orig_path, synset_id)
    if not os.path.isdir(synset_fpath) or '.' in synset_id:
        continue
    total_objects += len(os.listdir(synset_fpath))
start_time = time.time()
cur_iter = -1

for synset_id in sorted(os.listdir(shapenet_orig_path)):
    synset_fpath = os.path.join(shapenet_orig_path, synset_id)
    if not os.path.isdir(synset_fpath) or '.' in synset_id:
        continue
    for model_idx, source_id in enumerate(sorted(os.listdir(synset_fpath))):
        cur_iter += 1
        if cur_iter < 11646: continue
        custom_progress_bar(cur_iter, total_objects, start_time)
        print('---------------------------'+str(synset_id)+'::::'+str(source_id)+'-------------------------------------')
        save_synset_folder = os.path.join(output_dir, synset_id)
        if not os.path.exists(save_synset_folder):
            os.makedirs(save_synset_folder)

        save_fpath = os.path.join(save_synset_folder, source_id)
        if not os.path.exists(save_fpath):
            os.mkdir(save_fpath)

        cad_path = os.path.join(shapenet_orig_path, synset_id, source_id)
        obj_fpath = os.path.join(cad_path, 'models', 'model_normalized.obj')

        if not os.path.exists(obj_fpath):
            continue

        # # don't render again
        # rgb42path = os.path.join(save_fpath,'rgb_'+str(41)+'.png')
        # if os.path.exists(rgb42path):
        #     print('--already rendered-'+str(source_id)+'-------------------------------------')
        #     continue

        scale = get_norm_info(obj_fpath)

        # reset blender
        bproc.clean_up()

        # load object
        obj = bproc.loader.load_shapenet(shapenet_orig_path, synset_id, source_id, move_object_origin=False)
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", model_idx)

        # set up lights around the object
        light_energy = 60
        light_scale = 3
        lights = [bproc.types.Light() for _ in location]

        for idx, loc in enumerate(location):
            # set light
            lights[idx].set_type("POINT")
            lights[idx].set_location([light_scale*v for v in loc])
            lights[idx].set_energy(light_energy)

            # compute rotation based on vector going from location towards the location of object
            rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() - loc)
            # add homog cam pose based on location and rotation
            cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

        bproc.renderer.set_max_amount_of_samples(50)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())

        for idx, loc in enumerate(location):
            if idx <= 1: continue
            # save rgb images
            color_bgr_0 = data["colors"][idx]
            color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
            cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

            # save masks
            mask_0 = data["nocs"][idx][..., -1]
            cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)

            # save nocs
            xyz_0 = 2*(data["nocs"][idx][..., :3] - 0.5)
            # xyz need to rotate 90 degree to match CAD
            rot90 = np.array([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]])
            h, w = xyz_0.shape[0], xyz_0.shape[1]

            xyz_0 = ((rot90 @ xyz_0.reshape(-1, 3).T).T).reshape(h, w, 3)
            np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))