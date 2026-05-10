"""
NYCU Computer Vision 2026 HW3 - Model definition and training logic.
Training script for cell instance segmentation with Mask R-CNN.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from utils import read_maskfile



# Configuration
TRAIN_DIR = "./hw3-data-release/train"
NUM_CLASSES = 5  # background + 4 cell classes
BATCH_SIZE = 2
EPOCHS = 30
VAL_SPLIT = 0.2
MODEL_PATH = "model.pth"
BEST_MODEL_PATH = "model_best.pth"
EVAL_EVERY = 1
BASE_LR = 1e-3
FINETUNE_LR = 1e-4
FREEZE_EPOCHS = 5
FIXED_SIZE = 440
MAX_GRAD_NORM = 5.0
EARLY_STOP_PATIENCE = 8


def get_model(num_classes):
    """Build a Mask R-CNN model with ImageNet-pretrained ResNet-50 backbone."""
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_ch = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, num_classes)
    return model


class CellDataset(Dataset):
    """Dataset for loading cell images and instance-level targets."""

    def __init__(self, sample_dirs, augment=False):
        self.sample_dirs = [Path(d) for d in sample_dirs]
        self.augment = augment

    def __len__(self):
        if self.augment:
            return len(self.sample_dirs) * 2
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        base_len = len(self.sample_dirs)
        sample_idx = idx % base_len if self.augment else idx
        apply_aug = self.augment and idx >= base_len

        sample_dir = self.sample_dirs[sample_idx]
        image_id = torch.tensor([sample_idx], dtype=torch.int64)

        # Load RGBA image -> RGB float32 [0, 1].
        img = read_maskfile(str(sample_dir / "image.tif"))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        else:
            img = img[..., :3]
        img = img.astype(np.float32) / 255.0

        # Load all class masks upfront so augmentations can be applied jointly.
        raw_class_masks = {}
        for class_id in range(1, 5):
            mask_path = sample_dir / f"class{class_id}.tif"
            if not mask_path.exists():
                continue
            raw_class_masks[class_id] = read_maskfile(str(mask_path))

        # Augmented half gets one geometric transform:
        # horizontal or vertical flip.
        if apply_aug:
            aug_type = np.random.randint(0, 2)
            if aug_type == 0:
                img = img[:, ::-1, :].copy()
                raw_class_masks = {
                    key: value[:, ::-1].copy()
                    for key, value in raw_class_masks.items()
                }
            else:
                img = img[::-1, :, :].copy()
                raw_class_masks = {
                    key: value[::-1, :].copy()
                    for key, value in raw_class_masks.items()
                }

        # Fixed-size resize for stable throughput.
        target_w, target_h = FIXED_SIZE, FIXED_SIZE
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img = np.array(
            img_pil.resize((target_w, target_h), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        raw_class_masks = {
            class_id: np.array(
                Image.fromarray(class_mask.astype(np.float32)).resize(
                    (target_w, target_h), Image.NEAREST
                )
            )
            for class_id, class_mask in raw_class_masks.items()
        }

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        boxes, masks, labels = [], [], []
        for class_id, class_mask in raw_class_masks.items():
            # Each unique non-zero value is one cell instance
            for inst_id in np.unique(class_mask):
                if inst_id == 0:
                    continue
                binary = (class_mask == inst_id).astype(np.uint8)
                rows = np.where(binary.any(axis=1))[0]
                cols = np.where(binary.any(axis=0))[0]
                if rows.size == 0 or cols.size == 0:
                    continue
                x1, x2 = int(cols[0]), int(cols[-1])
                y1, y2 = int(rows[0]), int(rows[-1])
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                masks.append(binary)
                labels.append(class_id)

        if len(boxes) == 0:
            H, W = img_tensor.shape[1], img_tensor.shape[2]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": image_id,
            }
        else:
            areas = []
            for box in boxes:
                areas.append((box[2] - box[0]) * (box[3] - box[1]))
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": torch.tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": image_id,
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }
        return img_tensor, target


def collate_fn(batch):
    """Collate variable-length detection targets into batch tuples."""
    return tuple(zip(*batch))


def _encode_binary_mask(binary_mask):
    """Encode a binary mask to COCO RLE format."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def evaluate_ap50(model, val_loader, device):
    """Evaluate segmentation AP50 on the validation split."""
    model.eval()

    images = []
    annotations = []
    results = []
    ann_id = 1

    with torch.no_grad():
        for imgs, targets in val_loader:
            preds = model([img.to(device) for img in imgs])

            for img, target, pred in zip(imgs, targets, preds):
                h, w = int(img.shape[1]), int(img.shape[2])
                image_id = int(target["image_id"].item())

                images.append(
                    {
                        "id": image_id,
                        "width": w,
                        "height": h,
                        "file_name": f"{image_id}.tif",
                    }
                )

                gt_masks = target["masks"].numpy()
                gt_labels = target["labels"].numpy()
                gt_boxes = target["boxes"].numpy()
                for i in range(len(gt_labels)):
                    x1, y1, x2, y2 = gt_boxes[i]
                    bbox = [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ]
                    area = float((gt_masks[i] > 0).sum())
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(gt_labels[i]),
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": _encode_binary_mask(
                                gt_masks[i] > 0
                            ),
                        }
                    )
                    ann_id += 1

                boxes = pred["boxes"].detach().cpu().numpy()
                labels = pred["labels"].detach().cpu().numpy()
                scores = pred["scores"].detach().cpu().numpy()
                masks = pred["masks"].detach().cpu().numpy()[:, 0]

                for i in range(len(scores)):
                    x1, y1, x2, y2 = boxes[i]
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(labels[i]),
                            "bbox": [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                            ],
                            "score": float(scores[i]),
                            "segmentation": _encode_binary_mask(
                                masks[i] > 0.5
                            ),
                        }
                    )

    if len(images) == 0 or len(annotations) == 0:
        model.train()
        return 0.0

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": f"class{i}"} for i in range(1, 5)],
    }
    coco_gt.createIndex()

    if len(results) == 0:
        model.train()
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = float(coco_eval.stats[0])
    tqdm.write(f"  val AP50: {ap50:.4f}")
    model.train()
    return ap50


def train():
    """Train Mask R-CNN on the cell segmentation dataset."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    all_dirs = sorted(
        str(path) for path in Path(TRAIN_DIR).iterdir() if path.is_dir()
    )
    use_validation = VAL_SPLIT > 0
    n_val = (
        max(1, int(len(all_dirs) * VAL_SPLIT))
        if use_validation
        else 0
    )
    val_dirs, train_dirs = all_dirs[:n_val], all_dirs[n_val:]

    train_loader = DataLoader(
        CellDataset(train_dirs, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        CellDataset(val_dirs, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    model = get_model(NUM_CLASSES).to(device)

    # Phase 1: freeze backbone so heads stabilize before feature drift.
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Phase 1: backbone frozen for first {FREEZE_EPOCHS} epochs")

    # Only pass parameters that require gradients to the optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    # Cosine annealing over phase-1 epochs only; will be reset for phase 2
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FREEZE_EPOCHS,
        eta_min=1e-5,
    )

    print(
        f"Training on {device} | "
        f"{len(train_dirs)} train / {len(val_dirs)} val samples"
    )
    loss_history = []
    ap50_history = []
    best_ap50 = 0.0
    no_improve_epochs = 0
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        # Phase transition: unfreeze backbone at the start of phase 2.
        if epoch == FREEZE_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all parameters at a lower LR.
            optimizer = optim.SGD(
                model.parameters(),
                lr=FINETUNE_LR,
                momentum=0.9,
                weight_decay=5e-4,
            )
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(EPOCHS - FREEZE_EPOCHS),
                eta_min=1e-6,
            )
            tqdm.write(
                f"Phase 2: backbone unfrozen, LR reset to {FINETUNE_LR}"
            )

        model.train()
        total_loss = 0.0
        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
            leave=True,
        )
        for imgs, targets in batch_bar:
            imgs = [img.to(device) for img in imgs]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(imgs, targets)
            if any(
                torch.isnan(loss_value).any()
                for loss_value in loss_dict.values()
            ):
                debug_ids = [
                    int(target["image_id"].item())
                    for target in targets
                ]
                debug_paths = [
                    str(train_dirs[image_id])
                    for image_id in debug_ids
                ]
                clean_loss_dict = {
                    key: (
                        float(value.detach().cpu())
                        if not torch.isnan(value).any()
                        else "nan"
                    )
                    for key, value in loss_dict.items()
                }
                raise RuntimeError(
                    f"NaN loss detected at epoch {epoch + 1}. "
                    f"batch_sample_ids={debug_ids}, "
                    f"batch_sample_paths={debug_paths}, "
                    f"losses={clean_loss_dict}"
                )
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += losses.item()
            batch_bar.set_postfix(loss=f"{losses.item():.4f}")
        lr_scheduler.step()
        avg = total_loss / max(len(train_loader), 1)
        loss_history.append(avg)
        current_ap50 = None
        if use_validation and (epoch + 1) % EVAL_EVERY == 0:
            current_ap50 = evaluate_ap50(model, val_loader, device)
            ap50_history.append(current_ap50)
            if current_ap50 > best_ap50:
                best_ap50 = current_ap50
                no_improve_epochs = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                tqdm.write(
                    f"  ** New best AP50: {best_ap50:.4f} "
                    f"-> saved {BEST_MODEL_PATH}"
                )
            else:
                no_improve_epochs += 1
            epoch_bar.set_postfix(
                avg_loss=f"{avg:.4f}",
                ap50=f"{current_ap50:.4f}",
            )
            tqdm.write(
                f"Epoch {epoch + 1}/{EPOCHS} - avg_loss: {avg:.4f} "
                f"- val_AP50: {current_ap50:.4f}"
            )
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                tqdm.write(
                    "Early stopping: no AP50 improvement for "
                    f"{EARLY_STOP_PATIENCE} evaluations."
                )
                break
        else:
            if not use_validation:
                ap50_history.append(None)
            epoch_bar.set_postfix(avg_loss=f"{avg:.4f}")
            if use_validation:
                tqdm.write(
                    f"Epoch {epoch + 1}/{EPOCHS} - avg_loss: {avg:.4f}"
                )
            else:
                tqdm.write(
                    f"Epoch {epoch + 1}/{EPOCHS} - avg_loss: {avg:.4f} "
                    "- full training mode (no validation)"
                )

        # Always keep latest weights for fine-tuning/resume.
        torch.save(model.state_dict(), MODEL_PATH)
        if not use_validation:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        torch.cuda.empty_cache()

    torch.save(loss_history, "loss_history_finetune_2.pt")
    torch.save(ap50_history, "ap50_history_finetune_2.pt")

    print(f"Model saved as {MODEL_PATH}")


if __name__ == "__main__":
    train()
