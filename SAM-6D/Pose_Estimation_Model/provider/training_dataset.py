
import os
import sys
import json
import cv2
import trimesh
import numpy as np
import h5py

import torch
import torchvision.transforms as transforms

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa

from data_utils import (
    load_im,
    io_load_gt,
    io_load_masks,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    get_random_rotation,
    get_bbox,
    load_data_FoundationPose,
    transform_point_cloud,
    combine_transformations,
)

from augmentation_utils import (
    DepthAugmentation,
    DepthGaussianNoiseTransform,
    DepthCorrelatedGaussianNoiseTransform,
    DepthMissingTransform,
    DepthDropoutTransform,
    DepthEllipseDropoutTransform,
    DepthEllipseNoiseTransform,
    DepthBlurTransform,
)

from augmentation_utils import (
    MaskAugmentation,
    MaskDialateTransform,
    MaskBBoxFillTransform,
    MaskMissingTransform,
    MaskEllipseDropoutTransform,
    MaskLineSplit,
)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

class Dataset():
    def __init__(self, cfg, num_img_per_epoch=-1):
        self.cfg = cfg

        self.data_dir = cfg.data_dir
        self.num_img_per_epoch = num_img_per_epoch
        self.min_visib_px = cfg.min_px_count_visib
        self.min_pts_count = cfg.min_pts_count
        self.min_visib_frac = cfg.min_visib_fract
        self.augment_depth = cfg.augment_depth
        self.augment_mask = cfg.augment_mask
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.shift_range = cfg.shift_range
        self.img_size = cfg.img_size
        self.n_sample_observed_point = cfg.n_sample_observed_point
        self.n_sample_model_point = cfg.n_sample_model_point
        self.n_sample_template_point = cfg.n_sample_template_point
        self.single_view_pr = cfg.single_view_pr


        self.data_paths = [
            os.path.join('MegaPose-GSO', 'train_pbr_web'),
            os.path.join('MegaPose-ShapeNetCore', 'train_pbr_web'),
            os.path.join('FoundationPose-GSO', 'train_pbr'),
            os.path.join('FoundationPose-Objaverse', 'train_pbr'),
            os.path.join('MyPose-GSO', 'train_pbr'),
        ]
        self.model_paths = [
            os.path.join(self.data_dir, 'MegaPose-GSO', 'google_scanned_objects'),
            os.path.join(self.data_dir, 'MegaPose-ShapeNetCore', 'shapenetcorev2'),
            os.path.join(self.data_dir, 'FoundationPose-GSO', 'google_scanned_objects'),
            os.path.join(self.data_dir, 'FoundationPose-Objaverse', 'objaverse'),
            os.path.join(self.data_dir, 'MyPose-GSO', 'google_scanned_objects')
        ]
        self.templates_paths = [
            os.path.join(self.data_dir, 'MegaPose-GSO', 'templates'),
            os.path.join(self.data_dir, 'MegaPose-ShapeNetCore', 'templates'),
            os.path.join(self.data_dir, 'FoundationPose-GSO', 'templates'),
            os.path.join(self.data_dir, 'FoundationPose-Objaverse', 'templates'),
            os.path.join(self.data_dir, 'MyPose-GSO', 'templates'),
        ]
        
        self.dataset_paths = []

        # # get pathes to images for MegaPose train data
        # for f in self.data_paths[:2]:
        #     with open(os.path.join(self.data_dir, f, 'key_to_shard.json')) as fr:
        #         key_shards = json.load(fr)

        #         for k in key_shards.keys():
        #             path_name = os.path.join(f, "shard-" + f"{key_shards[k]:06d}", k)
        #             self.dataset_paths.append(path_name)

        
        # # get pathes to images for FoundationPose train data
        # for f in self.data_paths[2:4]:
        #     train_pbr_path = os.path.join(self.data_dir, f)
        #     for shard in os.listdir(train_pbr_path):
        #         shard_path = os.path.join(train_pbr_path, shard)
        #         for scene in os.listdir(shard_path):
        #             scene_path = os.path.join(shard_path, scene)
        #             for subscene in os.listdir(scene_path):
        #                 if os.path.isdir(os.path.join(scene_path, subscene)):
        #                     path_name = os.path.join(f, shard, scene, subscene)
        #                     self.dataset_paths.append(path_name)

        # get pathes to images for MyPose train data
        for f in self.data_paths[4:5]:
            train_pbr_path = os.path.join(self.data_dir, f)
            for shard in os.listdir(train_pbr_path):
                shard_path = os.path.join(train_pbr_path, shard)
                for scene in os.listdir(shard_path):
                    scene_path = os.path.join(shard_path, scene)
                    for subscene in os.listdir(scene_path):
                        if os.path.isdir(os.path.join(scene_path, subscene)):
                            path_name = os.path.join(f, shard, scene, subscene, "RenderProduct_Replicator")
                            self.dataset_paths.append(path_name)
                            path_name = os.path.join(f, shard, scene, subscene, "RenderProduct_Replicator_01")
                            self.dataset_paths.append(path_name)


        self.length = len(self.dataset_paths)
        print('Total {} images .....'.format(self.length))


        with open(os.path.join(self.data_dir, self.data_paths[0], 'gso_models.json')) as fr:
            self.model_info = [json.load(fr)]
        with open(os.path.join(self.data_dir, self.data_paths[1], 'shapenet_models.json')) as fr:
            self.model_info.append(json.load(fr))

        # gdrnpp aug 
        aug_code = (
            "Sequential(["
            "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
            "Sometimes(0.4, GaussianBlur((0., 3.))),"
            "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
            "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
            "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
            "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
            "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
            "Sometimes(0.3, Invert(0.2, per_channel=True)),"
            "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
            "Sometimes(0.5, Multiply((0.6, 1.4))),"
            "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
            "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
            "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
            "], random_order=True)"
            # cosy+aae
        )
        self.color_augmentor = eval(aug_code)
        self.depth_augmentor = DepthAugmentation(p=0.8,
            transform=[
                DepthAugmentation(p=0.3, transform=DepthBlurTransform()),
                DepthAugmentation(p=0.3, transform=DepthEllipseDropoutTransform()),
                DepthAugmentation(p=0.3, transform=DepthGaussianNoiseTransform()),
                DepthAugmentation(p=0.3, transform=DepthMissingTransform()),
            ]
        )
        self.mask_augmentor = MaskAugmentation(p=0.8,
            transform=[
                MaskAugmentation(p=0.3, transform=MaskBBoxFillTransform()),
                MaskAugmentation(p=0.3, transform=MaskDialateTransform()),
                MaskAugmentation(p=0.3, transform=MaskLineSplit()),
                MaskAugmentation(p=0.3, transform=MaskEllipseDropoutTransform()),
                MaskAugmentation(p=0.3, transform=MaskMissingTransform()),
            ]
        )

        self.template_mask_augmentor = MaskAugmentation(p=0.8,
            transform=[
                MaskAugmentation(p=0.3, transform=MaskLineSplit()),
                MaskAugmentation(p=0.3, transform=MaskEllipseDropoutTransform()),
                MaskAugmentation(p=0.3, transform=MaskMissingTransform()),
            ]
        )
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.length if self.num_img_per_epoch == -1 else self.num_img_per_epoch

    def reset(self):
        if self.num_img_per_epoch == -1:
            self.num_img_per_epoch = self.length

        num_img = self.length
        if num_img <= self.num_img_per_epoch:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch, replace=False)


    def __getitem__(self, index):
        while True:  # return valid data for train
            processed_data = self.read_data(self.img_idx[index])
            if processed_data is None:
                index = self._rand_another(index)
                continue
            return processed_data

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def read_data(self, index):
        # index = 100
        path_head = self.dataset_paths[index]
        dataset_auth = path_head.split('/')[0].split('-')[0]

        if dataset_auth == "MegaPose":
            return self.read_data_MegaPose(index)
        elif dataset_auth == "FoundationPose":
            return self.read_data_FoundationPose(index)
        elif dataset_auth == "MyPose":
            return self.read_data_MyPose(index)
        else:
            raise "data path should be eithder from MegaPose or FoundationPose"

    def augment_data_FoundationPose(self, cam_param_0, mask_0, depth_0, rgb_0, n_point, mask_augmentor):
         # mask
        if np.sum(mask_0) == 0:
            return None, None, None
        if self.augment_mask:
            mask_0 = mask_0.astype(np.float32)
            mask_0 = mask_augmentor(mask_0)
            if np.sum(mask_0>0) == 0:
                return None, None, None
        bbox_0 = get_bbox(mask_0>0)
        y1_0,y2_0,x1_0,x2_0 = bbox_0
        mask_0 = mask_0[y1_0:y2_0, x1_0:x2_0]

        # depth
        if self.augment_depth:
            # my depth aug is in mm
            depth_0 = self.depth_augmentor(depth_0 * 1000.0)/1000.0

        # choose valid depth & seg
        depth_0_cropped = depth_0[y1_0:y2_0, x1_0:x2_0]
        valid_mask = (mask_0 > 0) & (depth_0_cropped > 0)
        choose_0 = np.nonzero(valid_mask.flatten())[0]
        if len(choose_0) < self.min_pts_count:
            return None, None, None

        # lift points
        K_0 = cam_param_0["K"]
        pts_0 = get_point_cloud_from_depth(depth_0, K_0, [y1_0, y2_0, x1_0, x2_0])
        pts_0 = pts_0.reshape(-1, 3)[choose_0, :]

        # sample points
        if len(choose_0) <= n_point:
            choose_idx_0 = np.random.choice(np.arange(len(choose_0)), n_point)
        else:
            choose_idx_0 = np.random.choice(np.arange(len(choose_0)), n_point, replace=False)
        choose_0 = choose_0[choose_idx_0]
        pts_0 = pts_0[choose_idx_0]

        # rgb
        rgb_0 = rgb_0[..., ::-1][y1_0:y2_0, x1_0:x2_0, :]
        if np.random.rand() < 0.8:
            rgb_0 = self.color_augmentor.augment_image(rgb_0)
        if self.rgb_mask_flag:
            rgb_0 = rgb_0 * (mask_0[:,:,None]>0).astype(np.uint8)
        rgb_0 = cv2.resize(rgb_0, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb_0 = self.transform(np.array(rgb_0))
        rgb_choose_0 = get_resize_rgb_choose(choose_0, [y1_0, y2_0, x1_0, x2_0], self.img_size)

        return pts_0, rgb_0, rgb_choose_0


    def read_data_MyPose(self, index):
        path_head = self.dataset_paths[index]
        dataset_auth = path_head.split('/')[0].split('-')[0]
        dataset_type = path_head.split('/')[0].split('-')[1]
        view_idx = 0 if path_head.split('/')[-1] == "RenderProduct_Replicator" else 1
        upper_path_head = os.path.join(path_head, "..")
        data = load_data_FoundationPose(self.data_dir, upper_path_head, self.min_visib_frac, self.min_visib_px, False)
        if data is None:
            return None

        # gt rot and translation 
        target_R = data["target_R"]
        target_t = data["target_t"]
        target_s = data["target_s"]

        # get view data
        if view_idx:
            cam_param = data["cam_param_1"]
            mask = data["mask_1"]
            depth = data["depth_1"]
            rgb = data["rgb_1"]
        else:
            cam_param = data["cam_param"]
            mask = data["mask"]
            depth = data["depth"]
            rgb = data["rgb"]

        # augment data
        pts, rgb, rgb_choose = self.augment_data_FoundationPose(
            cam_param, mask, depth, rgb, self.n_sample_observed_point, self.mask_augmentor)
        if pts is None:
            return None

        # get rel R and t
        target_R, target_t = combine_transformations(cam_param["cam_in_world"], target_R, target_t)


        # get templates using obj_key
        tem_idx0, tem_idx1 = self.get_tem_idxs()
        obj_key = data['obj_key']
        tem1_rgb, tem1_choose, tem1_pts = self._get_template_MyPose(dataset_type, obj_key, tem_idx0)
        tem2_rgb, tem2_choose, tem2_pts = self._get_template_MyPose(dataset_type, obj_key, tem_idx1)
        if tem1_rgb is None or tem2_rgb is None:
            return None

        # adjust template pts for sim scale
        gso_path = self.model_paths[0]
        obj_fpath = os.path.join(gso_path, 'models_normalized', obj_key, 'meshes', 'model.obj')
        org_fpath = os.path.join(gso_path, 'models_orig', obj_key, 'meshes', 'model.obj')
        if not os.path.exists(obj_fpath) or not os.path.exists(org_fpath):
            print("did not find obj")
            print(obj_fpath)
            print(org_fpath)
            return None
        
        def get_norm_info(mesh_path):
            mesh = trimesh.load(mesh_path, force='mesh')

            model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
            model_points = model_points.astype(np.float32)

            min_value = np.min(model_points, axis=0)
            max_value = np.max(model_points, axis=0)

            radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

            return 1/(2*radius)

        from pathlib import Path
        import numpy as np
        from collections import defaultdict

        def get_orig_scale(mesh_path):
            # Read the entire OBJ file as text and split into lines.
            text = Path(mesh_path).read_text()
            lines = text.splitlines()

            # Group lines by the first token (e.g. "v", "f", etc.).
            elements = defaultdict(list)
            for line in lines:
                parts = line.split()
                if parts:  # ensure the line is not empty
                    elements[parts[0]].append(line)

            # Parse vertex positions.
            vertices = []
            for line in elements["v"]:
                parts = line.split()
                # parts[0] is "v", so we convert the rest to floats.
                vertices.append(list(map(float, parts[1:])))
            vertices = np.array(vertices)

            # Parse face lines to get vertex indices.
            # This code assumes faces are triangles.
            face_vertex_ids = []
            for face in elements["f"]:
                face_parts = face.split()
                # face_parts[0] is "f"; face_parts[1:4] are the three vertices.
                for n_idx in (1, 2, 3):
                    # Each vertex element is of the form "v_index/vt_index/vn_index"
                    vertex_index = face_parts[n_idx].split("/")[0]
                    face_vertex_ids.append(int(vertex_index))
            # Convert from OBJ's 1-indexing to Python's 0-indexing.
            face_vertex_ids = np.array(face_vertex_ids) - 1

            # Select only the vertices used in faces.
            positions = vertices[face_vertex_ids]
            
            # Compute the bounding box of the positions.
            xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
            ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
            zmin, zmax = positions[:, 2].min(), positions[:, 2].max()

            # The scale is defined as half the maximum extent (largest dimension) of the bounding box.
            scale = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0

            return scale


        tem_scale = get_norm_info(obj_fpath)
        orig_scale = get_orig_scale(org_fpath)

        print(tem_scale * 2)
        print(1/orig_scale)
        print(target_s)

        # tem1_pts *= orig_scale/(tem_scale*2)*target_s
        # tem2_pts *= orig_scale/(tem_scale*2)*target_s


        # rotation aug
        rand_R = get_random_rotation()
        tem1_pts = tem1_pts @ rand_R
        tem2_pts = tem2_pts @ rand_R
        target_R = target_R @ rand_R

        # translation aug
        add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        target_t = target_t + add_t[0]
        add_t = add_t + 0.001*np.random.randn(pts.shape[0], 3)
        pts = np.add(pts, add_t)

        ret_dict = {
            'pts': torch.FloatTensor(pts),
            'rgb': torch.FloatTensor(rgb),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
            'tem1_rgb': torch.FloatTensor(tem1_rgb),
            'tem1_choose': torch.IntTensor(tem1_choose).long(),
            'tem1_pts': torch.FloatTensor(tem1_pts),
            'tem2_rgb': torch.FloatTensor(tem2_rgb),
            'tem2_choose': torch.IntTensor(tem2_choose).long(),
            'tem2_pts': torch.FloatTensor(tem2_pts),
            'K': torch.FloatTensor(cam_param["K"]),
        }
        return ret_dict

    def get_tem_idxs(self):
        tem_idx0, tem_idx1 = 0,1 # change later if have more than two templates
        r = np.random.rand()
        if r < self.single_view_pr / 2:
            tem_idx1 = tem_idx0
        elif r < self.single_view_pr:
            tem_idx0 = tem_idx1
        return tem_idx0, tem_idx1

    def read_data_FoundationPose(self, index):
        path_head = self.dataset_paths[index]
        dataset_auth = path_head.split('/')[0].split('-')[0]
        dataset_type = path_head.split('/')[0].split('-')[1]
        data = load_data_FoundationPose(self.data_dir, path_head, self.min_visib_frac, self.min_visib_px)
        if data is None:
            return None

        # gt rot and translation 
        target_R = data["target_R"]
        target_t = data["target_t"]

        # get ref view data
        cam_param_0 = data["cam_param"]
        mask_0 = data["mask"]
        depth_0 = data["depth"]
        rgb_0 = data["rgb"]
        pts_0, rgb_0, rgb_choose_0 = self.augment_data_FoundationPose(
            cam_param_0, mask_0, depth_0, rgb_0, self.n_sample_template_point, self.template_mask_augmentor)
        if pts_0 is None:
            return None

        # move ref_0 to canonical pose
        pts_0 = transform_point_cloud(pts_0, cam_param_0["cam_in_world"])
        pts_0 = (pts_0 - target_t[None, :]) @ target_R
    
        # get second view data
        cam_param_1 = data["cam_param_1"]
        mask_1 = data["mask_1"]
        depth_1 = data["depth_1"]
        rgb_1 = data["rgb_1"]
        pts_1, rgb_1, rgb_choose_1 = self.augment_data_FoundationPose(
            cam_param_1, mask_1, depth_1, rgb_1, self.n_sample_observed_point, self.mask_augmentor)
        if pts_1 is None:
            return None

        # get rel R and t
        target_R, target_t = combine_transformations(cam_param_1["cam_in_world"], target_R, target_t)

        # rotation aug
        rand_R = get_random_rotation()
        pts_0 = pts_0 @ rand_R
        target_R = target_R @ rand_R

        # translation aug
        add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        target_t = target_t + add_t[0]
        add_t = add_t + 0.001*np.random.randn(pts_1.shape[0], 3)
        pts_1 = np.add(pts_1, add_t)


        ret_dict = {
            'pts': torch.FloatTensor(pts_1),
            'rgb': torch.FloatTensor(rgb_1),
            'rgb_choose': torch.IntTensor(rgb_choose_1).long(),
            'tem1_pts': torch.FloatTensor(pts_0),
            'tem1_rgb': torch.FloatTensor(rgb_0),
            'tem1_choose': torch.IntTensor(rgb_choose_0).long(),
            'tem2_pts': torch.FloatTensor(pts_0),
            'tem2_rgb': torch.FloatTensor(rgb_0),
            'tem2_choose': torch.IntTensor(rgb_choose_0).long(),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
            'K': torch.FloatTensor(cam_param_1["K"]),
        }
        return ret_dict

    def read_data_MegaPose(self, index):
        path_head = self.dataset_paths[index]
        dataset_auth = path_head.split('/')[0].split('-')[0]
        dataset_type = path_head.split('/')[0].split('-')[1]
        if not self._check_path(os.path.join(self.data_dir, path_head)):
            return None

        # gt_info
        gt_info = io_load_gt(open(os.path.join(self.data_dir, path_head+'.gt_info.json'), 'rb'))
        valid_idx = []
        for k, item in enumerate(gt_info):
            if item['px_count_valid'] >= self.min_visib_px and item['visib_fract'] >= self.min_visib_frac:
                # de baias training dataset
                if item['visib_fract'] > 0.95 and np.random.rand() > 0.1:
                    continue
                valid_idx.append(k)
        if len(valid_idx) == 0:
            return None
        num_instance = len(valid_idx)
        valid_idx = valid_idx[np.random.randint(0, num_instance)]
        gt_info = gt_info[valid_idx]
        # bbox = gt_info['bbox_visib']
        # x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]

        # gt
        gt = io_load_gt(open(os.path.join(self.data_dir, path_head+'.gt.json'), 'rb'))[valid_idx]
        obj_id = gt['obj_id']
        target_R = np.array(gt['cam_R_m2c']).reshape(3,3).astype(np.float32)
        target_t = np.array(gt['cam_t_m2c']).reshape(3).astype(np.float32) / 1000.0

        # camera
        camera = json.load(open(os.path.join(self.data_dir, path_head+'.camera.json'), 'rb'))
        K = np.array(camera['cam_K']).reshape(3,3)

        # template
        tem_idx0, tem_idx1 = self.get_tem_idxs()
        tem1_rgb, tem1_choose, tem1_pts = self._get_template(dataset_type, obj_id, tem_idx0)
        tem2_rgb, tem2_choose, tem2_pts = self._get_template(dataset_type, obj_id, tem_idx1)
        if tem1_rgb is None or tem2_rgb is None:
            return None

        # mask
        mask = io_load_masks(open(os.path.join(self.data_dir, path_head+'.mask_visib.json'), 'rb'))[valid_idx]
        if np.sum(mask) == 0:
            return None
        if self.augment_mask:
            mask = mask.astype(np.float32)
            mask = self.mask_augmentor(mask)
            if np.sum(mask>0) == 0:
                return None

        bbox = get_bbox(mask>0)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # depth
        depth = load_im(os.path.join(self.data_dir, path_head+'.depth.png')).astype(np.float32)
        if self.augment_depth:
            depth = self.depth_augmentor(depth)
        depth = depth * camera['depth_scale'] / 1000.0
        pts = get_point_cloud_from_depth(depth, K, [y1, y2, x1, x2])
        pts = pts.reshape(-1, 3)[choose, :]

        # select and sample
        if len(choose) < self.min_pts_count:
            return None

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # rgb
        rgb = load_im(os.path.join(self.data_dir, path_head+'.rgb.jpg')).astype(np.uint8)
        rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
        if np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        # rotation aug
        rand_R = get_random_rotation()
        tem1_pts = tem1_pts @ rand_R
        tem2_pts = tem2_pts @ rand_R
        target_R = target_R @ rand_R

        # translation aug
        add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        target_t = target_t + add_t[0]
        add_t = add_t + 0.001*np.random.randn(pts.shape[0], 3)
        pts = np.add(pts, add_t)

        ret_dict = {
            'pts': torch.FloatTensor(pts),
            'rgb': torch.FloatTensor(rgb),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
            'tem1_rgb': torch.FloatTensor(tem1_rgb),
            'tem1_choose': torch.IntTensor(tem1_choose).long(),
            'tem1_pts': torch.FloatTensor(tem1_pts),
            'tem2_rgb': torch.FloatTensor(tem2_rgb),
            'tem2_choose': torch.IntTensor(tem2_choose).long(),
            'tem2_pts': torch.FloatTensor(tem2_pts),
            'K': torch.FloatTensor(K),
        }
        return ret_dict

    def _get_template_MyPose(self, dataset_type, obj_key, tem_index=1):
        if dataset_type == 'GSO':
            file_base = os.path.join(
                self.templates_paths[0],
                obj_key,
            )
        elif dataset_type == 'Objaverse':
            raise NotImplementedError

        rgb_path = os.path.join(file_base, 'rgb_'+str(tem_index)+'.png')
        xyz_path = os.path.join(file_base, 'xyz_'+str(tem_index)+'.npy')
        mask_path = os.path.join(file_base, 'mask_'+str(tem_index)+'.png')
        if not os.path.exists(rgb_path):
            return None, None, None

        # mask
        mask = load_im(mask_path).astype(np.uint8) == 255

        # augment mask
        if self.augment_mask:
            mask = mask.astype(np.float32)
            total_visible = np.sum(mask>0)
            mask = self.template_mask_augmentor(mask)
            #check if mask is still useful
            #check min vsible fraction
            now_visible = np.sum(mask>0)
            if now_visible < self.min_visib_px:
                return None, None, None
            if now_visible/total_visible < self.min_visib_frac:
                return None, None, None
            
        bbox = get_bbox(mask)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]

        # rgb
        rgb = load_im(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]

        # augment image
        if np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        choose = mask.astype(np.float32).flatten().nonzero()[0]
        # apply the same min_pts_count to templates
        if len(choose) < self.min_pts_count:
            return None, None, None

        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        # xyz
        xyz = np.load(xyz_path).astype(np.float32)[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :] * 0.1

        # augment xyz
        if self.augment_depth and np.random.rand() < 0.8:
            mean_norm = np.mean(np.linalg.norm(xyz, axis=1))
            noise_scale = 0.01 * mean_norm
            noise = np.random.normal(scale=noise_scale, size=xyz.shape)
            xyz += noise

        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return rgb, choose, xyz


    def _get_template(self, type, obj_id, tem_index=1):
        if type == 'GSO':
            info = self.model_info[0][obj_id]
            assert info['obj_id'] == obj_id
            file_base = os.path.join(
                self.templates_paths[0],
                info['gso_id'],
            )

        elif type == 'ShapeNetCore':
            info = self.model_info[1][obj_id]
            assert info['obj_id'] == obj_id
            file_base = os.path.join(
                self.templates_paths[1],
                info['shapenet_synset_id'],
                info['shapenet_source_id'],
            )

        rgb_path = os.path.join(file_base, 'rgb_'+str(tem_index)+'.png')
        xyz_path = os.path.join(file_base, 'xyz_'+str(tem_index)+'.npy')
        mask_path = os.path.join(file_base, 'mask_'+str(tem_index)+'.png')
        if not os.path.exists(rgb_path):
            return None, None, None

        # mask
        mask = load_im(mask_path).astype(np.uint8) == 255

        # augment mask
        if self.augment_mask:
            mask = mask.astype(np.float32)
            total_visible = np.sum(mask>0)
            mask = self.template_mask_augmentor(mask)
            #check if mask is still useful
            #check min vsible fraction
            now_visible = np.sum(mask>0)
            if now_visible < self.min_visib_px:
                return None, None, None
            if now_visible/total_visible < self.min_visib_frac:
                return None, None, None
            
        bbox = get_bbox(mask)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]

        # rgb
        rgb = load_im(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]

        # augment image
        if np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        choose = mask.astype(np.float32).flatten().nonzero()[0]
        # apply the same min_pts_count to templates
        if len(choose) < self.min_pts_count:
            return None, None, None

        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        # xyz
        xyz = np.load(xyz_path).astype(np.float32)[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :] * 0.1

        # augment xyz
        if self.augment_depth and np.random.rand() < 0.8:
            mean_norm = np.mean(np.linalg.norm(xyz, axis=1))
            noise_scale = 0.01 * mean_norm
            noise = np.random.normal(scale=noise_scale, size=xyz.shape)
            xyz += noise

        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return rgb, choose, xyz

    def _check_path(self, path_head):
        keys = [
            '.camera.json',
            '.depth.png',
            '.gt_info.json',
            '.gt.json',
            '.mask_visib.json',
            '.rgb.jpg'
        ]

        for k in keys:
            if not os.path.exists(path_head + k):
                return False
        return True
