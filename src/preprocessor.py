"""
Image preprocessing utilities for the SAM segmentation pipeline.

Handles image resizing, normalization, automatic point prompt generation
from edge/contour detection, and grid-based point sampling for dense
segmentation workflows.
"""

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocessing pipeline for images before SAM inference.

    Provides resizing with aspect ratio preservation, pixel normalization,
    automatic object detection for prompt generation, and configurable
    grid-based point sampling.
    """

    def __init__(self, max_image_size: int = 2048):
        """Initialize the preprocessor.

        Args:
            max_image_size: Maximum dimension (width or height) for input images.
                Images exceeding this size are downscaled while preserving aspect ratio.
        """
        self.max_image_size = max_image_size

    def resize_for_sam(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> Tuple[np.ndarray, float]:
        """Resize an image while maintaining its aspect ratio.

        The longest side is scaled to target_size. If the image is already
        within limits, it is returned unchanged.

        Args:
            image: Input image array (H x W x C).
            target_size: Target dimension for the longest side. Defaults to
                self.max_image_size.
            interpolation: OpenCV interpolation method.

        Returns:
            Tuple of (resized_image, scale_factor). The scale factor can be
            used to map coordinates back to the original image space.
        """
        if target_size is None:
            target_size = self.max_image_size

        h, w = image.shape[:2]
        longest = max(h, w)

        if longest <= target_size:
            return image.copy(), 1.0

        scale = target_size / longest
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return resized, scale

    def normalize(
        self,
        image: np.ndarray,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Normalize pixel values to zero-mean unit-variance.

        Args:
            image: Input image array (H x W x 3) with uint8 or float values.
            mean: Per-channel mean for normalization. Defaults to ImageNet values.
            std: Per-channel standard deviation. Defaults to ImageNet values.

        Returns:
            Normalized float32 image array.
        """
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        normalized = (img - mean_arr) / std_arr
        return normalized

    def auto_detect_objects(
        self,
        image: np.ndarray,
        min_contour_area: int = 500,
        max_contour_area: Optional[int] = None,
        canny_low: int = 50,
        canny_high: int = 150,
        dilate_iterations: int = 2,
    ) -> List[dict]:
        """Detect object candidates using edge detection and contour analysis.

        Applies Canny edge detection followed by morphological dilation to
        close gaps, then finds external contours. For each contour, a
        centroid point prompt and bounding box are computed.

        Args:
            image: Input image array (H x W x 3), BGR or RGB.
            min_contour_area: Minimum contour area to be considered an object.
            max_contour_area: Maximum contour area. Defaults to 80% of image area.
            canny_low: Lower threshold for Canny edge detection.
            canny_high: Upper threshold for Canny edge detection.
            dilate_iterations: Number of dilation iterations to close edge gaps.

        Returns:
            List of dictionaries, each containing:
                - 'point': [x, y] centroid coordinate for a point prompt
                - 'bbox': [x1, y1, x2, y2] bounding box
                - 'area': contour area in pixels
                - 'contour': the raw OpenCV contour array
        """
        h, w = image.shape[:2]
        if max_contour_area is None:
            max_contour_area = int(h * w * 0.8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, canny_low, canny_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=dilate_iterations)

        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area or area > max_contour_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            x, y, bw, bh = cv2.boundingRect(contour)

            objects.append({
                "point": [cx, cy],
                "bbox": [x, y, x + bw, y + bh],
                "area": int(area),
                "contour": contour,
            })

        objects.sort(key=lambda o: o["area"], reverse=True)
        return objects

    def generate_grid_points(
        self,
        image_width: int,
        image_height: int,
        points_per_side: int = 32,
        offset_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a uniform grid of point prompts for dense segmentation.

        Creates an evenly spaced grid covering the image. Each point is
        assigned a positive foreground label.

        Args:
            image_width: Width of the target image in pixels.
            image_height: Height of the target image in pixels.
            points_per_side: Number of points along each dimension.
            offset_ratio: Offset from the cell boundary as a fraction of
                cell size (0.5 places points at cell centers).

        Returns:
            Tuple of (point_coords, point_labels) where point_coords is
            an (N, 2) array of [x, y] coordinates and point_labels is
            an (N,) array of ones (all foreground).
        """
        cell_w = image_width / points_per_side
        cell_h = image_height / points_per_side

        points = []
        for row in range(points_per_side):
            for col in range(points_per_side):
                x = int((col + offset_ratio) * cell_w)
                y = int((row + offset_ratio) * cell_h)

                x = min(x, image_width - 1)
                y = min(y, image_height - 1)
                points.append([x, y])

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.ones(len(points), dtype=np.int32)

        return point_coords, point_labels

    def generate_adaptive_points(
        self,
        image: np.ndarray,
        num_points: int = 64,
        edge_weight: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point prompts concentrated near edges and high-contrast regions.

        Combines a uniform grid with edge-density-weighted sampling to place
        more points near likely object boundaries.

        Args:
            image: Input image (H x W x 3).
            num_points: Total number of points to generate.
            edge_weight: Fraction of points placed near detected edges.
                The remainder are drawn from a uniform grid.

        Returns:
            Tuple of (point_coords, point_labels).
        """
        h, w = image.shape[:2]

        num_edge_points = int(num_points * edge_weight)
        num_grid_points = num_points - num_edge_points

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), 30, 100)

        edge_coords = np.column_stack(np.where(edges > 0))
        edge_points = []

        if len(edge_coords) > 0 and num_edge_points > 0:
            if len(edge_coords) >= num_edge_points:
                indices = np.random.choice(len(edge_coords), num_edge_points, replace=False)
            else:
                indices = np.random.choice(len(edge_coords), num_edge_points, replace=True)

            for idx in indices:
                y_coord, x_coord = edge_coords[idx]
                edge_points.append([int(x_coord), int(y_coord)])

        grid_side = max(1, int(math.sqrt(num_grid_points)))
        grid_coords, _ = self.generate_grid_points(w, h, points_per_side=grid_side)
        grid_points = grid_coords[:num_grid_points].tolist()

        all_points = edge_points + grid_points
        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.ones(len(all_points), dtype=np.int32)

        return point_coords, point_labels

    def prepare_image(
        self,
        image: Union[str, np.ndarray],
        resize: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Full preprocessing pipeline: load, resize, and return metadata.

        Args:
            image: File path or numpy array (BGR).
            resize: Whether to apply size limits.

        Returns:
            Tuple of (processed_image, metadata) where metadata includes
            original dimensions, scale factor, and final dimensions.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img = image.copy()

        original_h, original_w = img.shape[:2]

        if resize:
            img, scale = self.resize_for_sam(img)
        else:
            scale = 1.0

        final_h, final_w = img.shape[:2]

        metadata = {
            "original_width": original_w,
            "original_height": original_h,
            "final_width": final_w,
            "final_height": final_h,
            "scale_factor": scale,
        }

        return img, metadata

    def scale_points_to_original(
        self,
        points: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Map point coordinates from resized image space back to original image space.

        Args:
            points: (N, 2) array of [x, y] coordinates in the resized image.
            scale_factor: Scale factor that was used to resize the image.

        Returns:
            (N, 2) array of [x, y] coordinates in the original image space.
        """
        if scale_factor == 1.0:
            return points.copy()
        return (points / scale_factor).astype(np.float32)

    def scale_box_to_original(
        self,
        box: List[int],
        scale_factor: float,
    ) -> List[int]:
        """Map bounding box coordinates back to the original image space.

        Args:
            box: [x1, y1, x2, y2] in resized image coordinates.
            scale_factor: Scale factor used during resizing.

        Returns:
            [x1, y1, x2, y2] in original image coordinates.
        """
        if scale_factor == 1.0:
            return box.copy()
        return [int(coord / scale_factor) for coord in box]
