"""
Adopted from Megapose Augmentation Code
"""

# Standard Library
import dataclasses
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third Party
import cv2
import numpy as np
import PIL
import torch
from PIL import ImageEnhance, ImageFilter
from torchvision.datasets import ImageFolder

from data_utils import get_bbox



class DepthTransform(object):
    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        depth = self._transform_depth(depth)
        return depth

class DepthAugmentation(DepthTransform):
    def __init__(
        self,
        transform: Union[DepthTransform, List["DepthAugmentation"]],
        p: float = 1.0,
    ):
        self.p = p
        self.transform = transform

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if random.random() <= self.p:
            if isinstance(self.transform, list):
                for transform_ in self.transform:
                    depth = transform_(depth)
            else:
                depth = self.transform(depth)
        return depth

class DepthGaussianNoiseTransform(DepthTransform):
    """Adds random Gaussian noise to the depth image."""

    def __init__(self, std_dev: float = 0.02):
        self.std_dev = std_dev

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        noise = np.random.normal(scale=self.std_dev, size=depth.shape)
        depth[depth > 0] += noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        return depth


class DepthCorrelatedGaussianNoiseTransform(DepthTransform):
    """Adds random Gaussian noise to the depth image."""

    def __init__(
        self,
        std_dev: float = 0.01,
        gp_rescale_factor_min: float = 15.0,
        gp_rescale_factor_max: float = 40.0,
    ):
        self.std_dev = std_dev
        self.gp_rescale_factor_min = gp_rescale_factor_min
        self.gp_rescale_factor_max = gp_rescale_factor_max
        self.gp_rescale_factor_bounds = [gp_rescale_factor_min, gp_rescale_factor_max]

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        depth = np.copy(depth)
        rescale_factor = np.random.uniform(
            low=self.gp_rescale_factor_min,
            high=self.gp_rescale_factor_max,
        )

        small_H, small_W = (np.array([H, W]) / rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=self.std_dev, size=(small_H, small_W))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        depth[depth > 0] += additive_noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        return depth


class DepthMissingTransform(DepthTransform):
    """Randomly drop-out parts of the depth image."""

    def __init__(self, max_missing_fraction: float = 0.2, debug: bool = False):
        self.max_missing_fraction = max_missing_fraction
        self.debug = debug

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        v_idx, u_idx = np.where(depth > 0)
        if not self.debug:
            missing_fraction = np.random.uniform(0, self.max_missing_fraction)
        else:
            missing_fraction = self.max_missing_fraction
        dropout_ids = np.random.choice(
            np.arange(len(u_idx)), int(missing_fraction * len(u_idx)), replace=False
        )
        depth[v_idx[dropout_ids], u_idx[dropout_ids]] = 0
        return depth


class DepthDropoutTransform(DepthTransform):
    """Set the entire depth image to zero."""

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.zeros_like(depth)
        return depth


class DepthEllipseDropoutTransform(DepthTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
    ) -> None:
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    @staticmethod
    def generate_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params["ellipse_dropout_mean"])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(
            nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
        )
        # Shape: [num_ellipses_to_dropout x 2]
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        y_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        return x_radii, y_radii, angles, dropout_centers

    @staticmethod
    def dropout_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> np.ndarray:
        """Randomly drop a few ellipses in the image for robustness.

        Adapted from:
        https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        This is adapted from the DexNet 2.0 code:
        https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L53


        @param depth_img: a [H x W] set of depth z values
        """

        depth_img = depth_img.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img, noise_params=noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            depth_img = cv2.ellipse(
                depth_img,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=0,
                thickness=-1,
            )

        return depth_img

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = self.dropout_random_ellipses(depth, self._noise_params)
        return depth


class DepthEllipseNoiseTransform(DepthTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
        std_dev: float = 0.01,
    ) -> None:
        self.std_dev = std_dev
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth_img = depth
        depth_aug = depth_img.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img, noise_params=self._noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        additive_noise = np.random.normal(loc=0.0, scale=self.std_dev, size=x_radii.shape)

        # Dropout ellipses
        noise = np.zeros_like(depth)
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            noise = cv2.ellipse(
                noise,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=additive_noise[i],
                thickness=-1,
            )

        depth_aug[depth > 0] += noise[depth > 0]
        depth = depth_aug

        return depth


class DepthBlurTransform(DepthTransform):
    def __init__(self, factor_interval: Tuple[int, int] = (3, 7)):
        self.factor_interval = factor_interval

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        k = random.randint(*self.factor_interval)
        depth = cv2.blur(depth, (k, k))
        return depth


# mask augmentation utils #

class MaskTransform(object):
    DROP = 0.0
    FILL = 1.0
    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        mask = self._transform_mask(mask)
        return mask

class MaskAugmentation(MaskTransform):
    def __init__(
        self,
        transform: Union[MaskTransform, List["MaskAugmentation"]],
        p: float = 1.0,
    ):
        self.p = p
        self.transform = transform

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        
        if random.random() <= self.p:
            if isinstance(self.transform, list):
                for transform_ in self.transform:
                    mask = transform_(mask)
            else:
                mask = self.transform(mask)
        return mask

class MaskDialateTransform(MaskTransform):
    def __init__(self, size: Tuple[int, int] = (3, 3), iterations: int = 4):
        self.size = size
        self.iter = iterations

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, self.size), iterations=self.iter)
        return mask

class MaskBBoxFillTransform(MaskTransform):
    """fill the whole bbox of a mask"""

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        bbox = get_bbox(mask>0)
        y1,y2,x1,x2 = bbox
        mask[y1:y2, x1:x2] = MaskTransform.FILL
        return mask


class MaskMissingTransform(MaskTransform):
    """Randomly drop-out parts of the mask."""

    def __init__(self, max_missing_fraction: float = 0.2):
        self.max_missing_fraction = max_missing_fraction

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        v_idx, u_idx = np.where(mask > 0)
        missing_fraction = np.random.uniform(0, self.max_missing_fraction)
        dropout_ids = np.random.choice(
            np.arange(len(u_idx)), int(missing_fraction * len(u_idx)), replace=False
        )
        mask[v_idx[dropout_ids], u_idx[dropout_ids]] = MaskTransform.DROP
        return mask


class MaskEllipseDropoutTransform(MaskTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
    ) -> None:
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    @staticmethod
    def generate_random_ellipses(
        mask: np.ndarray, noise_params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params["ellipse_dropout_mean"])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(mask > 0)).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(
            nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
        )
        # Shape: [num_ellipses_to_dropout x 2]
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        y_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        return x_radii, y_radii, angles, dropout_centers

    @staticmethod
    def dropout_random_ellipses(
        mask: np.ndarray, noise_params: Dict[str, float]
    ) -> np.ndarray:
        """Randomly drop a few ellipses in the mask for robustness.

        Adapted from:
        https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        This is adapted from the DexNet 2.0 code:
        https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L53


        @param mask: a [H x W] segmentation mask
        """
        mask = mask.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            mask, noise_params=noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            mask = cv2.ellipse(
                mask,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=MaskTransform.DROP,
                thickness=-1,
            )

        return mask

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = self.dropout_random_ellipses(mask, self._noise_params)
        return mask

class MaskLineSplit(MaskTransform):
    """
    randomly splitting the mask into two parts.
    This ensures a max 0.5 split fraction
    """
    @staticmethod
    def split_and_drop_smallest(mask: np.ndarray) -> np.ndarray:
        """
        Splits a segmentation mask into two parts by:
        1. Choosing a random point on the mask.
        2. Choosing a random angle to define a line through that point.
        3. Dropping the smaller side of the mask.
        
        This ensures that at most 50% of the mask area is removed.
        
        Args:
            mask (np.ndarray): 2D segmentation mask (binary or multi-class).
        
        Returns:
            np.ndarray: Augmented mask with the smaller part removed.
        """
        # Find all non-zero points
        nonzero_points = np.column_stack(np.where(mask > 0))
        if nonzero_points.shape[0] == 0:
            # No points to split
            return mask

        # Randomly choose a point from the mask
        chosen_idx = random.randint(0, nonzero_points.shape[0] - 1)
        y0, x0 = nonzero_points[chosen_idx]  # nonzero_points is (row, col) = (y, x)

        # Choose a random angle
        theta = 2 * np.pi * np.random.random()
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Compute c for the line x*cos_t + y*sin_t = c
        c = x0 * cos_t + y0 * sin_t

        # Create a coordinate grid for the mask
        h, w = mask.shape
        y_grid, x_grid = np.indices((h, w))

        # Determine which side each pixel is on
        side_values = x_grid * cos_t + y_grid * sin_t
        side_mask = side_values > c

        # Count how many non-zero pixels fall on each side
        # We only consider non-zero pixels to determine which side is smaller
        nonzero_y, nonzero_x = np.nonzero(mask)
        nonzero_side_values = nonzero_x * cos_t + nonzero_y * sin_t
        side1_count = np.sum(nonzero_side_values > c)
        side2_count = np.sum(nonzero_side_values <= c)

        # Determine which side is smaller
        if side1_count < side2_count:
            # Drop side1 (where side_values > c)
            mask[side_values > c] = MaskTransform.DROP
        else:
            # Drop side2 (where side_values <= c)
            mask[side_values <= c] = MaskTransform.DROP

        return mask

    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = MaskLineSplit.split_and_drop_smallest(mask)
        return mask