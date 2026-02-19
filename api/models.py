"""
Pydantic models for the SAM segmentation API.

Defines request and response schemas for point-prompted segmentation,
box-prompted segmentation, automatic segmentation, background removal,
and object extraction endpoints.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class PointPrompt(BaseModel):
    """A single point prompt for SAM segmentation.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        label: Point label (1 = foreground/positive, 0 = background/negative).
    """

    x: int = Field(..., ge=0, description="X coordinate in pixels")
    y: int = Field(..., ge=0, description="Y coordinate in pixels")
    label: int = Field(1, ge=0, le=1, description="1 for foreground, 0 for background")


class BoxPrompt(BaseModel):
    """A bounding box prompt for SAM segmentation.

    Attributes:
        x1: Left edge x coordinate.
        y1: Top edge y coordinate.
        x2: Right edge x coordinate.
        y2: Bottom edge y coordinate.
    """

    x1: int = Field(..., ge=0, description="Left x coordinate")
    y1: int = Field(..., ge=0, description="Top y coordinate")
    x2: int = Field(..., ge=0, description="Right x coordinate")
    y2: int = Field(..., ge=0, description="Bottom y coordinate")

    @field_validator("x2")
    @classmethod
    def x2_must_be_greater_than_x1(cls, v, info):
        if "x1" in info.data and v <= info.data["x1"]:
            raise ValueError("x2 must be greater than x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_must_be_greater_than_y1(cls, v, info):
        if "y1" in info.data and v <= info.data["y1"]:
            raise ValueError("y2 must be greater than y1")
        return v

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class SegmentationRequest(BaseModel):
    """Request body for point-prompted segmentation.

    Attributes:
        points: List of point prompts with coordinates and labels.
        multimask_output: Whether to return multiple mask candidates.
        refine_mask: Whether to apply morphological refinement.
        return_largest: If True, only return the largest mask.
    """

    points: List[PointPrompt] = Field(..., min_length=1, description="List of point prompts")
    multimask_output: bool = Field(True, description="Return multiple mask candidates")
    refine_mask: bool = Field(True, description="Apply morphological mask refinement")
    return_largest: bool = Field(False, description="Return only the largest mask")


class BoxSegmentationRequest(BaseModel):
    """Request body for box-prompted segmentation.

    Attributes:
        box: Bounding box coordinates.
        multimask_output: Whether to return multiple mask candidates.
        refine_mask: Whether to apply morphological refinement.
    """

    box: BoxPrompt = Field(..., description="Bounding box prompt")
    multimask_output: bool = Field(False, description="Return multiple mask candidates")
    refine_mask: bool = Field(True, description="Apply morphological mask refinement")


class AutoSegmentationRequest(BaseModel):
    """Request body for automatic segmentation.

    Attributes:
        min_area: Minimum mask area in pixels.
        max_masks: Maximum number of masks to return.
        points_per_side: Grid density for automatic point generation.
    """

    min_area: Optional[int] = Field(None, ge=0, description="Minimum mask area in pixels")
    max_masks: Optional[int] = Field(None, ge=1, description="Maximum masks to return")
    points_per_side: int = Field(32, ge=4, le=128, description="Points per grid side")


class BackgroundRemovalRequest(BaseModel):
    """Request body for background removal.

    Attributes:
        feather_radius: Gaussian blur radius for edge softening.
        point: Optional foreground point to guide segmentation.
    """

    feather_radius: int = Field(3, ge=0, le=20, description="Edge feathering radius")
    point: Optional[PointPrompt] = Field(None, description="Optional foreground point prompt")


class MaskResult(BaseModel):
    """A single mask result from segmentation.

    Attributes:
        mask_index: Index of this mask in the results list.
        iou_score: Predicted IoU quality score from SAM.
        area: Mask area in pixels.
        bbox: Bounding box of the mask [x1, y1, width, height].
        stability_score: Mask stability score (auto segmentation only).
    """

    mask_index: int = Field(..., description="Mask index in the results")
    iou_score: float = Field(..., ge=0.0, le=1.0, description="Predicted IoU score")
    area: int = Field(..., ge=0, description="Mask area in pixels")
    bbox: Optional[List[int]] = Field(None, description="Bounding box [x, y, w, h]")
    stability_score: Optional[float] = Field(None, description="Stability score")


class SegmentationResponse(BaseModel):
    """Response from segmentation endpoints.

    Attributes:
        masks: List of mask results with quality metrics.
        num_masks: Total number of masks returned.
        image_width: Width of the input image.
        image_height: Height of the input image.
        processing_time_ms: Total processing time in milliseconds.
    """

    masks: List[MaskResult] = Field(..., description="List of mask results")
    num_masks: int = Field(..., ge=0, description="Number of masks returned")
    image_width: int = Field(..., ge=1, description="Input image width")
    image_height: int = Field(..., ge=1, description="Input image height")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")


class ExtractionRequest(BaseModel):
    """Request body for object extraction.

    Attributes:
        points: Optional point prompts. If not provided, auto-detection is used.
        box: Optional bounding box prompt.
        padding: Pixel padding around the extracted object.
    """

    points: Optional[List[PointPrompt]] = Field(None, description="Point prompts for target object")
    box: Optional[BoxPrompt] = Field(None, description="Bounding box prompt")
    padding: int = Field(10, ge=0, le=200, description="Padding around extracted object")

    @field_validator("points")
    @classmethod
    def at_least_one_prompt(cls, v, info):
        return v


class ErrorResponse(BaseModel):
    """Standard error response.

    Attributes:
        detail: Human-readable error message.
        error_code: Machine-readable error identifier.
    """

    detail: str = Field(..., description="Error message")
    error_code: str = Field("UNKNOWN_ERROR", description="Error code identifier")
