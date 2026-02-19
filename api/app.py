"""
FastAPI application for SAM-based image segmentation.

Provides endpoints for point-prompted segmentation, box-prompted
segmentation, automatic mask generation, background removal, and
object extraction with transparent output.
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from api.models import (
    AutoSegmentationRequest,
    BackgroundRemovalRequest,
    BoxPrompt,
    BoxSegmentationRequest,
    ExtractionRequest,
    MaskResult,
    PointPrompt,
    SegmentationRequest,
    SegmentationResponse,
)
from src.preprocessor import ImagePreprocessor
from src.sam_model import SAMSegmenter


_CONFIG_PATH = os.environ.get("SAM_CONFIG_PATH", "configs/config.yaml")

_segmenter: Optional[SAMSegmenter] = None
_preprocessor: Optional[ImagePreprocessor] = None


def _load_config() -> dict:
    """Load the application configuration from YAML."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SAM model at startup and release at shutdown."""
    global _segmenter, _preprocessor

    config = _load_config()
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processing", {})

    _segmenter = SAMSegmenter(
        model_type=model_cfg.get("model_type", "vit_l"),
        checkpoint=model_cfg.get("checkpoint", "sam_vit_l_0b3195.pth"),
        device=model_cfg.get("device", "cuda"),
    )

    _preprocessor = ImagePreprocessor(
        max_image_size=proc_cfg.get("max_image_size", 2048),
    )

    print("SAM segmentation service ready")
    yield
    print("Shutting down SAM service")


app = FastAPI(
    title="Segment Anything API",
    description="Image segmentation service powered by Meta's Segment Anything Model",
    version="1.0.0",
    lifespan=lifespan,
)


def _read_upload_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded file bytes to a BGR numpy image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file. Ensure the upload is a valid image.")
    return image


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a binary mask as a PNG byte string."""
    mask_img = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", mask_img)
    return buffer.tobytes()


def _pil_to_png_bytes(pil_image: Image.Image) -> bytes:
    """Encode a PIL RGBA image as PNG bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _mask_bbox(mask: np.ndarray):
    """Compute bounding box [x, y, w, h] from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, y1 = int(xs.min()), int(ys.min())
    return [x1, y1, int(xs.max()) - x1 + 1, int(ys.max()) - y1 + 1]


@app.post("/segment/points", response_model=SegmentationResponse, summary="Segment with point prompts")
async def segment_points(
    file: UploadFile = File(..., description="Image file to segment"),
    points_json: str = Form(..., description="JSON: [{\"x\": 100, \"y\": 200, \"label\": 1}, ...]"),
    multimask_output: bool = Form(True),
    refine_mask: bool = Form(True),
):
    """Segment an image using point prompts.

    Upload an image and provide point prompts as a JSON string. Each point
    has x/y coordinates and a label (1 = foreground, 0 = background).
    """
    import json
    contents = await file.read()
    image_bgr = _read_upload_image(contents)
    image_rgb = _bgr_to_rgb(image_bgr)
    h, w = image_rgb.shape[:2]

    try:
        raw_points = json.loads(points_json)
        points = [PointPrompt(**p) for p in raw_points]
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points JSON: {e}")

    coords = [[p.x, p.y] for p in points]
    labels = [p.label for p in points]

    start = time.perf_counter()
    results = _segmenter.segment_with_points(
        image_rgb, coords, labels, multimask_output=multimask_output
    )

    if refine_mask:
        for r in results:
            r["mask"] = _segmenter.refine_mask(r["mask"])
            r["area"] = int(r["mask"].sum())

    elapsed = (time.perf_counter() - start) * 1000

    mask_results = []
    for idx, r in enumerate(results):
        mask_results.append(MaskResult(
            mask_index=idx,
            iou_score=round(r["iou_score"], 4),
            area=r["area"],
            bbox=_mask_bbox(r["mask"]),
        ))

    return SegmentationResponse(
        masks=mask_results,
        num_masks=len(mask_results),
        image_width=w,
        image_height=h,
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/segment/box", response_model=SegmentationResponse, summary="Segment with bounding box")
async def segment_box(
    file: UploadFile = File(..., description="Image file to segment"),
    x1: int = Form(..., description="Left x coordinate"),
    y1: int = Form(..., description="Top y coordinate"),
    x2: int = Form(..., description="Right x coordinate"),
    y2: int = Form(..., description="Bottom y coordinate"),
    multimask_output: bool = Form(False),
    refine_mask: bool = Form(True),
):
    """Segment an image using a bounding box prompt.

    Provide the bounding box coordinates as form fields. The region within
    the box is segmented into a binary mask.
    """
    contents = await file.read()
    image_bgr = _read_upload_image(contents)
    image_rgb = _bgr_to_rgb(image_bgr)
    h, w = image_rgb.shape[:2]

    box = [x1, y1, x2, y2]

    start = time.perf_counter()
    results = _segmenter.segment_with_box(
        image_rgb, box, multimask_output=multimask_output
    )

    if refine_mask:
        for r in results:
            r["mask"] = _segmenter.refine_mask(r["mask"])
            r["area"] = int(r["mask"].sum())

    elapsed = (time.perf_counter() - start) * 1000

    mask_results = []
    for idx, r in enumerate(results):
        mask_results.append(MaskResult(
            mask_index=idx,
            iou_score=round(r["iou_score"], 4),
            area=r["area"],
            bbox=_mask_bbox(r["mask"]),
        ))

    return SegmentationResponse(
        masks=mask_results,
        num_masks=len(mask_results),
        image_width=w,
        image_height=h,
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/segment/auto", response_model=SegmentationResponse, summary="Automatic segmentation")
async def segment_auto(
    file: UploadFile = File(..., description="Image file to segment"),
    min_area: Optional[int] = Form(None, description="Minimum mask area in pixels"),
    max_masks: Optional[int] = Form(None, description="Maximum masks to return"),
):
    """Automatically segment all objects in an image.

    No prompts are required. The model generates masks for every detected
    region, filtered by minimum area and maximum count.
    """
    contents = await file.read()
    image_bgr = _read_upload_image(contents)
    image_rgb = _bgr_to_rgb(image_bgr)
    h, w = image_rgb.shape[:2]

    start = time.perf_counter()
    results = _segmenter.segment_everything(
        image_rgb, min_area=min_area, max_masks=max_masks
    )
    elapsed = (time.perf_counter() - start) * 1000

    mask_results = []
    for idx, r in enumerate(results):
        mask_results.append(MaskResult(
            mask_index=idx,
            iou_score=round(r.get("predicted_iou", 0.0), 4),
            area=r.get("area", 0),
            bbox=list(r.get("bbox", [0, 0, 0, 0])),
            stability_score=round(r.get("stability_score", 0.0), 4),
        ))

    return SegmentationResponse(
        masks=mask_results,
        num_masks=len(mask_results),
        image_width=w,
        image_height=h,
        processing_time_ms=round(elapsed, 2),
    )


@app.post("/background/remove", summary="Remove background from image")
async def remove_background(
    file: UploadFile = File(..., description="Image file"),
    feather_radius: int = Form(3, description="Edge feathering radius"),
    point_x: Optional[int] = Form(None, description="Foreground point X coordinate"),
    point_y: Optional[int] = Form(None, description="Foreground point Y coordinate"),
):
    """Remove the background from an uploaded image.

    Returns a PNG image with transparent background. Optionally provide
    a foreground point to guide which object is preserved.
    """
    contents = await file.read()
    image_bgr = _read_upload_image(contents)
    image_rgb = _bgr_to_rgb(image_bgr)

    point = [point_x, point_y] if point_x is not None and point_y is not None else None

    start = time.perf_counter()
    result_image = _segmenter.remove_background(
        image_rgb,
        feather_radius=feather_radius,
        point=point,
    )
    elapsed = (time.perf_counter() - start) * 1000

    png_bytes = _pil_to_png_bytes(result_image)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": str(round(elapsed, 2)),
            "Content-Disposition": "inline; filename=result.png",
        },
    )


@app.post("/extract", summary="Extract object as transparent PNG")
async def extract_object(
    file: UploadFile = File(..., description="Image file"),
    point_x: int = Form(..., description="Foreground point X coordinate"),
    point_y: int = Form(..., description="Foreground point Y coordinate"),
    padding: int = Form(10, description="Padding around extracted object"),
):
    """Extract a specific object from an image with a transparent background.

    Provide a point on the target object. The extracted object is tightly
    cropped with optional padding and returned as a PNG with alpha channel.
    """
    contents = await file.read()
    image_bgr = _read_upload_image(contents)
    image_rgb = _bgr_to_rgb(image_bgr)

    start = time.perf_counter()

    results = _segmenter.segment_with_points(
        image_rgb,
        [[point_x, point_y]],
        [1],
        multimask_output=True,
    )

    if not results:
        raise HTTPException(status_code=422, detail="No mask could be generated for the given point.")

    best_mask = _segmenter.get_largest_mask([r["mask"] for r in results])
    best_mask = _segmenter.refine_mask(best_mask)

    extracted = _segmenter.extract_object(image_rgb, best_mask, padding=padding)
    elapsed = (time.perf_counter() - start) * 1000

    png_bytes = _pil_to_png_bytes(extracted)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Processing-Time-Ms": str(round(elapsed, 2)),
            "X-Mask-Area": str(int(best_mask.sum())),
            "Content-Disposition": "inline; filename=extracted.png",
        },
    )
