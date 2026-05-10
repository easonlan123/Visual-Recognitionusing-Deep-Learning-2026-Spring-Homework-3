"""Inference script for cell instance segmentation predictions."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms

from train import FIXED_SIZE, NUM_CLASSES, get_model
from utils import encode_mask, read_maskfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TEST_IMG_DIR = "./hw3-data-release/test_release"
TEST_META = "./hw3-data-release/test_image_name_to_ids.json"
MODEL_PATH = "model_best.pth"
SCORE_THRESH = 0.05
NMS_IOU_THRESH = 0.3
MAX_DETS_PER_IMAGE = 100


def run_inference():
    """Run model inference on test images and write COCO-style results."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = get_model(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(TEST_META) as f:
        meta = json.load(f)
    name_to_id = {item["file_name"]: item["id"] for item in meta}

    test_dir = Path(TEST_IMG_DIR)
    results = []

    print(f"Running inference from {MODEL_PATH}...")
    for img_path in sorted(test_dir.glob("*.tif")):
        image_id = name_to_id.get(img_path.name)
        if image_id is None:
            print(
                f"  Warning: {img_path.name} not found in metadata, "
                "skipping"
            )
            continue

        img = read_maskfile(str(img_path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., :3]
        orig_h, orig_w = img.shape[:2]
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_resized = np.array(
            img_pil.resize((FIXED_SIZE, FIXED_SIZE), Image.BILINEAR)
        ).astype(np.float32)
        img_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1)

        with torch.no_grad():
            preds = model([img_tensor.to(device)])[0]

        boxes = preds["boxes"].detach().cpu()
        scores = preds["scores"].detach().cpu()
        labels = preds["labels"].detach().cpu()
        masks = preds["masks"].detach().cpu()

        # Filter by score threshold
        keep = scores >= SCORE_THRESH
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

        # Apply NMS per-class
        if len(scores) > 0:
            keep_idx = batched_nms(boxes, scores, labels, NMS_IOU_THRESH)
            if len(keep_idx) > MAX_DETS_PER_IMAGE:
                keep_idx = keep_idx[:MAX_DETS_PER_IMAGE]
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            masks = masks[keep_idx]

        scores = scores.numpy()
        labels = labels.numpy()
        masks = masks.numpy()

        for score, label, mask in zip(scores, labels, masks):
            binary = (
                (mask[0] > 0.5).astype(np.uint8)
                if mask.ndim == 3
                else (mask > 0.5).astype(np.uint8)
            )
            if binary.shape != (orig_h, orig_w):
                binary = np.array(
                    Image.fromarray(binary).resize(
                        (orig_w, orig_h),
                        Image.NEAREST,
                    ),
                    dtype=np.uint8,
                )
            if binary.sum() == 0:
                continue
            results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "segmentation": encode_mask(binary),
                "score": float(score),
            })

    with open("test-results.json", "w") as f:
        json.dump(results, f)
    print(f"Done. {len(results)} predictions written to test-results.json")


if __name__ == "__main__":
    run_inference()
