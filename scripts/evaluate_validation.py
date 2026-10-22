from pathlib import Path
import csv
import json
from statistics import mean

import kagglehub
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.arcade_dataset import ArcadeSegmentationDataset


# 1. Locate project files
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "unet_imagenet_best.pt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. Download public ARCADE dataset if not cached
dataset_root = Path(
    kagglehub.dataset_download("nirmalgaud/arcade-dataset")
)

val_dir = dataset_root / "arcade" / "syntax" / "val"

# 3. Load validation dataset
dataset = ArcadeSegmentationDataset(
    images_dir=val_dir / "images",
    annotation_file=val_dir / "annotations" / "val.json",
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

print(f"Validation images: {len(dataset)}")

# 4. Reconstruct the trained model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=1,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 5. Evaluate every image
results = []

with torch.inference_mode():
    for batch_index, (images, masks) in enumerate(loader):

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        predictions = probabilities >= 0.5
        targets = masks >= 0.5

        for j in range(images.shape[0]):
            pred = predictions[j]
            gt = targets[j]

            tp = int((pred & gt).sum().item())
            fp = int((pred & ~gt).sum().item())
            fn = int((~pred & gt).sum().item())

            dice_den = 2 * tp + fp + fn
            iou_den = tp + fp + fn

            dice = 2 * tp / dice_den if dice_den else 1.0
            iou = tp / iou_den if iou_den else 1.0

            precision = (
                tp / (tp + fp)
                if tp + fp else (1.0 if tp + fn == 0 else 0.0)
            )

            recall = (
                tp / (tp + fn)
                if tp + fn else 1.0
            )

            index = batch_index * loader.batch_size + j

            results.append({
                "filename": dataset.images[index]["file_name"],
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

        print(f"Evaluated {len(results)}/{len(dataset)} images")

# 6. Calculate average metrics
summary = {
    "model": "unet_imagenet_best",
    "split": "validation",
    "threshold": 0.5,
    "num_images": len(results),
    "mean_dice": mean(r["dice"] for r in results),
    "mean_iou": mean(r["iou"] for r in results),
    "mean_precision": mean(r["precision"] for r in results),
    "mean_recall": mean(r["recall"] for r in results),
}

# 7. Save per-image results
csv_path = RESULTS_DIR / "validation_image_metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# 8. Save summary
json_path = RESULTS_DIR / "validation_summary.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

# 9. Print summary
print("\nVALIDATION RESULTS")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFIVE LOWEST DICE SCORES")
for result in sorted(results, key=lambda r: r["dice"])[:5]:
    print(
        f"{result['filename']}: "
        f"Dice={result['dice']:.4f}, "
        f"Recall={result['recall']:.4f}"
    )

print("\nResults saved successfully.")