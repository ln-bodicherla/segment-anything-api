# Segment Anything Model (SAM) API Service

A production-ready API service for automated image segmentation using [Meta's Segment Anything Model](https://github.com/facebookresearch/segment-anything). Supports point prompts, bounding box prompts, and fully automatic mask generation. Built for practical applications including background removal, object isolation, and annotation generation.

## Features

- **Point-Prompted Segmentation** -- Provide positive and negative point prompts to segment specific objects with fine-grained control.
- **Box-Prompted Segmentation** -- Use bounding box coordinates to isolate objects within a defined region.
- **Automatic Mask Generation** -- Generate segmentation masks for every object in an image without manual prompts.
- **Background Removal** -- Remove image backgrounds and produce clean RGBA output with feathered edges.
- **Object Extraction** -- Extract individual objects as transparent PNG images ready for compositing.
- **Mask Refinement** -- Morphological post-processing to clean mask boundaries and remove artifacts.
- **COCO Export** -- Convert generated masks to COCO annotation format for dataset creation.

## Project Structure

```
segment-anything-api/
├── api/
│   ├── app.py
│   └── models.py
├── src/
│   ├── sam_model.py
│   └── preprocessor.py
├── configs/
│   └── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Model Checkpoint

```bash
mkdir -p weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O weights/sam_vit_l_0b3195.pth
```

Available checkpoints:
- `sam_vit_h_4b8939.pth` (2.4 GB) -- ViT-H, highest quality
- `sam_vit_l_0b3195.pth` (1.2 GB) -- ViT-L, balanced quality and speed
- `sam_vit_b_01ec64.pth` (375 MB) -- ViT-B, fastest inference

### Python Usage

```python
from src.sam_model import SAMSegmenter

segmenter = SAMSegmenter(model_type="vit_l", checkpoint="weights/sam_vit_l_0b3195.pth")

# Segment with point prompts
masks = segmenter.segment_with_points(
    image_path="photo.jpg",
    points=[[400, 300], [450, 350]],
    labels=[1, 1],  # 1 = positive, 0 = negative
)

# Background removal
rgba_image = segmenter.remove_background("portrait.jpg")
rgba_image.save("portrait_no_bg.png")

# Automatic segmentation
all_masks = segmenter.segment_everything("scene.jpg", min_area=500)
```

### Start the API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/segment/points` | Segment using point prompts |
| POST | `/segment/box` | Segment using bounding box |
| POST | `/segment/auto` | Automatic full-image segmentation |
| POST | `/background/remove` | Remove background from image |
| POST | `/extract` | Extract object as transparent PNG |

## Configuration

All model and processing parameters are configured through `configs/config.yaml`, including model type, automatic mask generator thresholds, image size limits, and background removal settings.

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU inference is supported but slow)
- See `requirements.txt` for full dependency list

## License

MIT License
