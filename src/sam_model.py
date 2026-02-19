"""
Core SAM segmentation engine.

Wraps Meta's Segment Anything Model to provide point-prompted segmentation,
box-prompted segmentation, automatic mask generation, background removal,
object extraction, and COCO-format mask export.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


_SUPPORTED_MODELS = {"vit_h", "vit_l", "vit_b"}


class SAMSegmenter:
    """High-level segmentation interface built on Segment Anything.

    Supports interactive point/box prompts as well as fully automatic mask
    generation. Includes utilities for mask refinement, background removal,
    object extraction, and COCO annotation conversion.
    """

    def __init__(
        self,
        model_type: str = "vit_l",
        checkpoint: str = "sam_vit_l_0b3195.pth",
        device: str = "cuda",
    ):
        """Initialize the SAM segmentation engine.

        Args:
            model_type: SAM backbone variant ('vit_h', 'vit_l', or 'vit_b').
            checkpoint: Path to the SAM weights file.
            device: Computation device ('cuda' or 'cpu').
        """
        if model_type not in _SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose from {_SUPPORTED_MODELS}")

        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"

        self.sam = None
        self.predictor: Optional[SamPredictor] = None
        self.mask_generator: Optional[SamAutomaticMaskGenerator] = None

        self._current_image: Optional[np.ndarray] = None
        self._image_set = False

        self.load_model()

    def load_model(self) -> None:
        """Load the SAM checkpoint and initialize the predictor and mask generator."""
        print(f"Loading SAM {self.model_type} from {self.checkpoint} on {self.device}")

        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=100,
        )

        print("SAM model loaded successfully")

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load an image from file path or validate an existing array.

        Returns the image in RGB format (as SAM expects).
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                img = image
            else:
                raise ValueError("Image array must be HxWx3 (RGB)")
        else:
            raise TypeError(f"Expected str or ndarray, got {type(image)}")
        return img

    def _set_image(self, image: np.ndarray) -> None:
        """Set the image in the SAM predictor (avoids redundant encoding)."""
        if self._current_image is not image:
            self.predictor.set_image(image)
            self._current_image = image
            self._image_set = True

    def segment_with_points(
        self,
        image: Union[str, np.ndarray],
        points: List[List[int]],
        labels: List[int],
        multimask_output: bool = True,
    ) -> List[Dict[str, Any]]:
        """Segment an image using point prompts.

        Args:
            image: Image file path or RGB numpy array.
            points: List of [x, y] coordinates for each prompt point.
            labels: List of labels for each point (1 = foreground, 0 = background).
            multimask_output: If True, return 3 mask candidates ranked by quality.

        Returns:
            List of dictionaries with keys: mask (ndarray), iou_score (float), area (int).
        """
        img = self._load_image(image)
        self._set_image(img)

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )

        results = []
        for i in range(masks.shape[0]):
            mask = masks[i].astype(np.uint8)
            results.append({
                "mask": mask,
                "iou_score": float(scores[i]),
                "area": int(mask.sum()),
            })

        results.sort(key=lambda r: r["iou_score"], reverse=True)
        return results

    def segment_with_box(
        self,
        image: Union[str, np.ndarray],
        box: List[int],
        multimask_output: bool = False,
    ) -> List[Dict[str, Any]]:
        """Segment an image using a bounding box prompt.

        Args:
            image: Image file path or RGB numpy array.
            box: Bounding box as [x1, y1, x2, y2] in pixel coordinates.
            multimask_output: If True, return multiple mask candidates.

        Returns:
            List of dictionaries with keys: mask, iou_score, area.
        """
        img = self._load_image(image)
        self._set_image(img)

        box_array = np.array(box, dtype=np.float32)

        masks, scores, logits = self.predictor.predict(
            box=box_array,
            multimask_output=multimask_output,
        )

        results = []
        for i in range(masks.shape[0]):
            mask = masks[i].astype(np.uint8)
            results.append({
                "mask": mask,
                "iou_score": float(scores[i]),
                "area": int(mask.sum()),
            })

        results.sort(key=lambda r: r["iou_score"], reverse=True)
        return results

    def segment_everything(
        self,
        image: Union[str, np.ndarray],
        min_area: Optional[int] = None,
        max_masks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run automatic mask generation on the full image.

        Args:
            image: Image file path or RGB numpy array.
            min_area: Minimum mask area in pixels. Masks smaller than this are discarded.
            max_masks: Maximum number of masks to return (sorted by area descending).

        Returns:
            List of SAM annotation dictionaries containing 'segmentation',
            'area', 'predicted_iou', 'stability_score', and 'bbox'.
        """
        img = self._load_image(image)
        masks = self.mask_generator.generate(img)

        if min_area is not None:
            masks = [m for m in masks if m["area"] >= min_area]

        masks.sort(key=lambda m: m["area"], reverse=True)

        if max_masks is not None:
            masks = masks[:max_masks]

        print(f"Automatic segmentation produced {len(masks)} masks")
        return masks

    def refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
        iterations: int = 2,
        close_holes: bool = True,
        remove_small_components: bool = True,
        min_component_area: int = 100,
    ) -> np.ndarray:
        """Apply morphological operations to clean a binary mask.

        Args:
            mask: Binary mask (H x W) with values 0 or 1.
            kernel_size: Size of the morphological structuring element.
            iterations: Number of morphological operation iterations.
            close_holes: Whether to apply morphological closing to fill holes.
            remove_small_components: Whether to remove small disconnected regions.
            min_component_area: Minimum area for a connected component to be kept.

        Returns:
            Refined binary mask.
        """
        refined = mask.copy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        if close_holes:
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

        if remove_small_components:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
            cleaned = np.zeros_like(refined)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= min_component_area:
                    cleaned[labels == label_id] = 1
            refined = cleaned

        return refined

    def extract_object(
        self,
        image: Union[str, np.ndarray],
        mask: np.ndarray,
        padding: int = 0,
    ) -> Image.Image:
        """Extract the segmented object with a transparent background.

        Args:
            image: Source image (file path or RGB numpy array).
            mask: Binary mask indicating the object region.
            padding: Pixel padding around the bounding box of the extracted object.

        Returns:
            PIL Image in RGBA mode with the object on a transparent background.
        """
        img = self._load_image(image)
        h, w = img.shape[:2]

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return Image.fromarray(rgba, "RGBA")

        x1 = max(0, int(xs.min()) - padding)
        y1 = max(0, int(ys.min()) - padding)
        x2 = min(w, int(xs.max()) + 1 + padding)
        y2 = min(h, int(ys.max()) + 1 + padding)

        cropped = rgba[y1:y2, x1:x2]
        return Image.fromarray(cropped, "RGBA")

    def remove_background(
        self,
        image: Union[str, np.ndarray],
        feather_radius: int = 3,
        point: Optional[List[int]] = None,
    ) -> Image.Image:
        """Remove the background from an image, returning an RGBA result.

        By default, segments the largest foreground object. A specific
        object can be targeted by providing a point prompt.

        Args:
            image: Image file path or RGB numpy array.
            feather_radius: Gaussian blur radius for softening mask edges.
            point: Optional [x, y] point on the foreground object to segment.

        Returns:
            PIL Image in RGBA mode with the background removed.
        """
        img = self._load_image(image)
        h, w = img.shape[:2]

        if point is not None:
            results = self.segment_with_points(img, [point], [1], multimask_output=True)
        else:
            center_x, center_y = w // 2, h // 2
            results = self.segment_with_points(
                img,
                [[center_x, center_y]],
                [1],
                multimask_output=True,
            )

        if not results:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = 255
            return Image.fromarray(rgba, "RGBA")

        mask = self.get_largest_mask([r["mask"] for r in results])

        mask = self.refine_mask(mask, kernel_size=5, iterations=2)

        alpha = (mask * 255).astype(np.uint8)
        if feather_radius > 0:
            ksize = feather_radius * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img
        rgba[:, :, 3] = alpha

        return Image.fromarray(rgba, "RGBA")

    def get_largest_mask(self, masks: List[np.ndarray]) -> np.ndarray:
        """Select the mask with the largest area from a list of candidates.

        Args:
            masks: List of binary masks (H x W arrays).

        Returns:
            The mask with the highest pixel count.
        """
        if not masks:
            raise ValueError("No masks provided")

        best_mask = masks[0]
        best_area = best_mask.sum()

        for mask in masks[1:]:
            area = mask.sum()
            if area > best_area:
                best_mask = mask
                best_area = area

        return best_mask

    def masks_to_coco(
        self,
        masks: List[Dict[str, Any]],
        image_id: int,
        image_width: int,
        image_height: int,
        category_id: int = 1,
        start_annotation_id: int = 1,
    ) -> Dict[str, Any]:
        """Convert a list of masks to COCO annotation format.

        Args:
            masks: List of mask dictionaries from segment_with_points,
                segment_with_box, or segment_everything.
            image_id: COCO image ID.
            image_width: Width of the source image.
            image_height: Height of the source image.
            category_id: Category ID to assign to all annotations.
            start_annotation_id: Starting ID for annotation numbering.

        Returns:
            COCO-format dictionary with 'images', 'annotations', and 'categories'.
        """
        from pycocotools import mask as mask_utils

        coco = {
            "images": [
                {
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                }
            ],
            "annotations": [],
            "categories": [
                {
                    "id": category_id,
                    "name": "object",
                    "supercategory": "none",
                }
            ],
        }

        ann_id = start_annotation_id
        for mask_data in masks:
            if "mask" in mask_data:
                binary_mask = mask_data["mask"].astype(np.uint8)
            elif "segmentation" in mask_data:
                binary_mask = mask_data["segmentation"].astype(np.uint8)
            else:
                continue

            fortran_mask = np.asfortranarray(binary_mask)
            rle = mask_utils.encode(fortran_mask)
            rle["counts"] = rle["counts"].decode("utf-8")

            area = int(binary_mask.sum())

            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0:
                continue
            x1, y1 = int(xs.min()), int(ys.min())
            bbox_w = int(xs.max()) - x1 + 1
            bbox_h = int(ys.max()) - y1 + 1

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygons = []
            for contour in contours:
                if contour.shape[0] >= 3:
                    polygon = contour.flatten().tolist()
                    polygons.append(polygon)

            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": polygons if polygons else rle,
                "area": area,
                "bbox": [x1, y1, bbox_w, bbox_h],
                "iscrowd": 0,
                "score": mask_data.get("iou_score", mask_data.get("predicted_iou", 1.0)),
            }

            coco["annotations"].append(annotation)
            ann_id += 1

        print(f"Generated {len(coco['annotations'])} COCO annotations for image {image_id}")
        return coco
