import os
import numpy as np
import json
import imageio
import cv2
import json

from PIL import Image

def load_im(path):
    """Loads an image from a file.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    if not os.path.isfile(path):
        return None
    im = imageio.imread(path)
    return im


def io_load_gt(
    gt_file,
    instance_ids=None,
):
    """Load ground truth from an I/O object.
    Instance_ids can be specified to load only a
    subset of object instances.

    :param gt_file: I/O object that can be read with json.load.
    :param instance_ids: List of instance ids.
    :return: List of ground truth annotations (one dict per object instance).
    """
    gt = json.load(gt_file)
    if instance_ids is not None:
        gt = [gt_n for n, gt_n in enumerate(gt) if n in instance_ids]
    gt = [_gt_as_numpy(gt_n) for gt_n in gt]
    return gt


def io_load_masks(
    mask_file,
    instance_ids=None
):
    """Load object masks from an I/O object.
    Instance_ids can be specified to apply RLE
    decoding to a subset of object instances contained
    in the file.

    :param mask_file: I/O object that can be read with json.load.
    :param masks_path: Path to json file.
    :return: a [N,H,W] binary array containing object masks.
    """
    masks_rle = json.load(mask_file)
    masks_rle = {int(k): v for k, v in masks_rle.items()}
    if instance_ids is None:
        instance_ids = masks_rle.keys()
        instance_ids = sorted(instance_ids)
    masks = np.stack([
        rle_to_binary_mask(masks_rle[instance_id])
        for instance_id in instance_ids])
    return masks


def _gt_as_numpy(gt):
    if 'cam_R_m2c' in gt.keys():
        gt['cam_R_m2c'] = \
        np.array(gt['cam_R_m2c'], np.float64).reshape((3, 3))
    if 'cam_t_m2c' in gt.keys():
        gt['cam_t_m2c'] = \
        np.array(gt['cam_t_m2c'], np.float64).reshape((3, 1))
    return gt


def rle_to_binary_mask(rle):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts = rle.get('counts')

    start = 0
    for i in range(len(counts)-1):
        start += counts[i] 
        end = start + counts[i+1] 
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


def get_point_cloud_from_depth(depth, K, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0,0], K[1,1], K[0,2], K[1,2]

    im_H, im_W = depth.shape
    xmap = np.array([[i for i in range(im_W)] for j in range(im_H)])
    ymap = np.array([[j for i in range(im_W)] for j in range(im_H)])

    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
        xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
        ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

    pt2 = depth.astype(np.float32)
    pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
    pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

    cloud = np.stack([pt0,pt1,pt2]).transpose((1,2,0))
    return cloud


def get_resize_rgb_choose(choose, bbox, img_size):
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    ratio_h = img_size / crop_h
    crop_w = cmax - cmin
    ratio_w = img_size / crop_w

    row_idx = choose // crop_w
    col_idx = choose % crop_w
    choose = (np.floor(row_idx * ratio_h) * img_size + np.floor(col_idx * ratio_w)).astype(np.int64)
    return choose


def get_bbox(label):
    img_width, img_length = label.shape
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    b = min(max(r_b, c_b), min(img_width, img_length))
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    rmin = center[0] - int(b / 2)
    rmax = center[0] + int(b / 2)
    cmin = center[1] - int(b / 2)
    cmax = center[1] + int(b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return [rmin, rmax, cmin, cmax]

def get_random_rotation():
    angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = np.array([
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]), np.cos(angles[2]),0],
        [0,0,1]
    ])
    return rand_rotation

def get_model_info(obj, return_color=False, sample_num=2048):
    if return_color:
        model_points, model_color, symmetry_flag = obj.get_item(return_color, sample_num)
        return (model_points, model_color, symmetry_flag)
    else:
        model_points,  symmetry_flag = obj.get_item()
        return (model_points, symmetry_flag)

def get_bop_depth_map(inst):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    try:
        depth = np.array(Image.open(os.path.join(data_folder, f'{scene_id:06d}', 'depth', f'{img_id:06d}.png'))) / 1000.0
    except:
        depth = np.array(Image.open(os.path.join(data_folder, f'{scene_id:06d}', 'depth', f'{img_id:06d}.tif'))) / 1000.0
    return depth

def get_bop_image(inst, bbox, img_size, mask=None):
    scene_id, img_id, data_folder = inst['scene_id'], inst['img_id'], inst['data_folder']
    rmin, rmax, cmin, cmax = bbox
    img_path = os.path.join(data_folder, f'{scene_id:06d}/')

    strs = [f'rgb/{img_id:06d}.jpg', f'rgb/{img_id:06d}.png', f'gray/{img_id:06d}.tif']
    for s in strs:
        if os.path.exists(os.path.join(img_path,s)):
            img_path = os.path.join(img_path,s)
            break

    rgb = load_im(img_path).astype(np.uint8)
    if len(rgb.shape)==2:
        rgb = np.concatenate([rgb[:,:,None], rgb[:,:,None], rgb[:,:,None]], axis=2)
    rgb = rgb[..., ::-1][rmin:rmax, cmin:cmax, :3]
    if mask is not None:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return rgb

# Utils to load FoundationPose dataset
def process_foundationpose_cam(camera_params):
    glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)
    W, H = camera_params["renderProductResolution"]
    world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
    cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
    world_in_cam = np.linalg.inv(cam_in_world)
    focal_length = camera_params["cameraFocalLength"]
    horiz_aperture = camera_params["cameraAperture"][0]
    vert_aperture = H / W * horiz_aperture
    focal_y = H * focal_length / vert_aperture
    focal_x = W * focal_length / horiz_aperture
    center_y = H * 0.5
    center_x = W * 0.5

    fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy

    camera = {"K":K,
              "world_in_cam":world_in_cam,
              "cam_in_world":cam_in_world}
    return camera

def transform_point_cloud(points, T):
    # Number of points
    N = points.shape[0]
    
    # Convert to homogeneous coordinates by appending a column of ones
    ones = np.ones((N, 1))
    points_homogeneous = np.hstack([points, ones])  # shape: (N, 4)
    
    # Apply the transformation matrix.
    # Note: We multiply from the left, so we need to transpose if using row vectors.
    transformed_homogeneous = (T @ points_homogeneous.T).T  # shape: (N, 4)
    
    # Convert back to 3D by dividing by the homogeneous coordinate, if necessary.
    # For an affine transformation, the 4th coordinate should be 1.
    # The following handles both cases.
    transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, np.newaxis]
    
    return transformed_points

def combine_transformations(T, target_R, target_t):
    # 1. Compute the translation part v = f(0)
    origin = np.zeros((1, 3))
    v = transform_point_cloud(origin, T)[0]
    
    # 2. Compute the linear part M by probing the action on basis vectors.
    M = np.zeros((3, 3))
    for i in range(3):
        # Create the i-th standard basis vector (as a 1x3 row vector)
        e = np.zeros((1, 3))
        e[0, i] = 1.0
        # The effect of T on e: f(e) = e*M + v, so row i of M is:
        M[i, :] = transform_point_cloud(e, T)[0] - v
    
    # 3. The combined rotation is:
    new_target_R = M @ target_R
    # 4. The combined translation is:
    new_target_t = (target_t - v) @ np.linalg.inv(M)
    
    return new_target_R, new_target_t

def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def load_npy(path):
    if not os.path.isfile(path):
        return None
    return np.load(path)

def convert_det_ids_to_seg_ids(det_ids, det_sem_map, seg_sem_map):
    # Build a lookup: from class name (value) to segmentation id (key)
    background = ["BACKGROUND", "UNLABELLED", "world_collision_box_collision_box_floor"]
    class_to_seg_id = {entry["class"]: seg_id for seg_id, entry in seg_sem_map.items()}
    
    seg_ids = []
    for det_id in det_ids:
        det_key = str(det_id)
        det_class = det_sem_map[det_key]["class"]

        if (det_class in class_to_seg_id) and (det_class not in background):
            seg_ids.append(int(class_to_seg_id[det_class]))
    return seg_ids

def load_data_FoundationPose(data_dir, path_head, min_visib_frac, min_visib_px):
    path_head = os.path.join(data_dir, path_head)
    states = load_json(os.path.join(path_head, "../states.json"))
    render_prod_replic_path = os.path.join(path_head, "RenderProduct_Replicator")
    render_prod_replic_path_1 = os.path.join(path_head, "RenderProduct_Replicator_01")
    if np.random.rand() < 0.5:
        render_prod_replic_path, render_prod_replic_path_1 = render_prod_replic_path_1, render_prod_replic_path

    rgb = load_im(os.path.join(render_prod_replic_path, "rgb/rgb_000000.png"))
    seg = load_im(os.path.join(render_prod_replic_path, "instance_segmentation/instance_segmentation_000000.png"))
    seg_ins_map = load_json(os.path.join(render_prod_replic_path, "instance_segmentation/instance_segmentation_mapping_000000.json"))
    seg_sem_map = load_json(os.path.join(render_prod_replic_path, "instance_segmentation/instance_segmentation_semantics_mapping_000000.json"))
    dis = load_npy(os.path.join(render_prod_replic_path, "distance_to_image_plane/distance_to_image_plane_000000.npy"))
    cam = load_json(os.path.join(render_prod_replic_path, "camera_params/camera_params_000000.json"))
    det_loose = load_npy(os.path.join(render_prod_replic_path, "bounding_box_2d_loose/bounding_box_2d_loose_000000.npy"))
    det_sem_map = load_json(os.path.join(render_prod_replic_path, "bounding_box_2d_loose/bounding_box_2d_loose_labels_000000.json"))

    rgb_1 = load_im(os.path.join(render_prod_replic_path_1, "rgb/rgb_000000.png"))
    seg_1 = load_im(os.path.join(render_prod_replic_path_1, "instance_segmentation/instance_segmentation_000000.png"))
    seg_ins_map_1 = load_json(os.path.join(render_prod_replic_path_1, "instance_segmentation/instance_segmentation_mapping_000000.json"))
    seg_sem_map_1 = load_json(os.path.join(render_prod_replic_path_1, "instance_segmentation/instance_segmentation_semantics_mapping_000000.json"))
    dis_1 = load_npy(os.path.join(render_prod_replic_path_1, "distance_to_image_plane/distance_to_image_plane_000000.npy"))
    cam_1 = load_json(os.path.join(render_prod_replic_path_1, "camera_params/camera_params_000000.json"))
    det_loose_1 = load_npy(os.path.join(render_prod_replic_path_1, "bounding_box_2d_loose/bounding_box_2d_loose_000000.npy"))
    det_sem_map_1 = load_json(os.path.join(render_prod_replic_path_1, "bounding_box_2d_loose/bounding_box_2d_loose_labels_000000.json"))

    # chaeck all files
    if any(x is None for x in [rgb, seg, seg_ins_map, seg_sem_map, dis, cam, det_loose, det_sem_map,
                             rgb_1, seg_1, seg_ins_map_1, seg_sem_map_1, dis_1, cam_1, det_loose_1, det_sem_map_1]):
        return None

    cam_param = process_foundationpose_cam(cam)
    cam_param_1 = process_foundationpose_cam(cam_1)
    
    occlusion_ratios = det_loose['occlusionRatio']
    visib_frac = 1.0 - occlusion_ratios
    mask = visib_frac > min_visib_frac
    selected_det_ids = det_loose['semanticId'][mask]
    selected_ids = convert_det_ids_to_seg_ids(selected_det_ids, det_sem_map, seg_sem_map)
    if len(selected_ids) == 0: return None
    unique_ids, counts = np.unique(seg, return_counts=True)
    count_dict = dict(zip(unique_ids, counts))
    final_ids = [
        oid for oid in selected_ids
        if count_dict.get(oid, 0) >= min_visib_px
    ]

    occlusion_ratios_1 = det_loose_1['occlusionRatio']
    visib_frac_1 = 1.0 - occlusion_ratios_1
    mask_1 = visib_frac_1 > min_visib_frac
    selected_det_ids_1 = det_loose_1['semanticId'][mask_1]
    selected_ids_1 = convert_det_ids_to_seg_ids(selected_det_ids_1, det_sem_map_1, seg_sem_map_1)
    if len(selected_ids_1) == 0: return None 
    unique_ids_1, counts_1 = np.unique(seg_1, return_counts=True)
    count_dict_1 = dict(zip(unique_ids_1, counts_1))
    final_ids_1 = [
        oid for oid in selected_ids_1
        if count_dict_1.get(oid, 0) >= min_visib_px
    ]

    valid_ids = np.intersect1d(final_ids_1, final_ids)
    num_valids = len(valid_ids)
    if num_valids == 0:
        return None
    obj_id = valid_ids[np.random.randint(0, num_valids)]
    obj_key = "_".join(seg_ins_map[str(obj_id)].split("/")[3].split("_")[1:])

    target_R = np.array(states["objects"][obj_key]["rotation_matrix"]).reshape(3,3).astype(np.float32)
    target_t = np.array(states["objects"][obj_key]["translation"]).reshape(3).astype(np.float32)

    mask   = (seg   == obj_id).astype(np.uint8)
    mask_1 = (seg_1 == obj_id).astype(np.uint8)

    depth = np.where(np.isinf(dis), 0, dis.astype(np.float32))
    depth_1 = np.where(np.isinf(dis_1), 0, dis_1.astype(np.float32))

    rgb   = rgb[..., :3].astype(np.uint8)
    rgb_1 = rgb_1[..., :3].astype(np.uint8)

    scene_data = {
        "rgb":rgb,
        "mask":mask,
        "depth":depth,
        "cam_param":cam_param,
        "rgb_1":rgb_1,
        "mask_1":mask_1,
        "depth_1":depth_1,
        "cam_param_1":cam_param_1,
        "target_R":target_R,
        "target_t":target_t,
        "obj_id":obj_id,
        "obj_key":obj_key
    }

    return scene_data