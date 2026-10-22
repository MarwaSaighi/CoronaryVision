from pathlib import Path
from statistics import mean
import csv

import kagglehub
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.arcade_dataset import ArcadeSegmentationDataset


# Locate the validation dataset
ROOT = Path(__file__).resolve().parents[1]

dataset_root = Path(
    kagglehub.dataset_download("nirmalgaud/arcade-dataset")
)

val_dir = dataset_root / "arcade" / "syntax" / "val"

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

# Load trained ImageNet U-Net
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=1,
)

checkpoint = torch.load(
    ROOT / "models" / "unet_imagenet_best.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Thresholds to compare
thresholds = [i / 10 for i in range(1, 10)]

scores = {
    t: {"dice": [], "precision": [], "recall": []}
    for t in thresholds
}

# Predict once per image batch, then test all thresholds
with torch.inference_mode():
    for batch_index, (images, masks) in enumerate(loader):

        probabilities = torch.sigmoid(model(images))
        gt = masks >= 0.5

        for threshold in thresholds:
            pred = probabilities >= threshold

            tp = (pred & gt).sum(dim=(1, 2, 3)).float()
            fp = (pred & ~gt).sum(dim=(1, 2, 3)).float()
            fn = (~pred & gt).sum(dim=(1, 2, 3)).float()

            dice_den = 2 * tp + fp + fn
            dice = torch.where(
                dice_den > 0,
                2 * tp / dice_den.clamp_min(1),
                torch.ones_like(tp),
            )

            precision = torch.where(
                tp + fp > 0,
                tp / (tp + fp).clamp_min(1),
                torch.where(
                    tp + fn > 0,
                    torch.zeros_like(tp),
                    torch.ones_like(tp),
                ),
            )

            recall = torch.where(
                tp + fn > 0,
                tp / (tp + fn).clamp_min(1),
                torch.ones_like(tp),
            )

            scores[threshold]["dice"].extend(dice.tolist())
            scores[threshold]["precision"].extend(precision.tolist())
            scores[threshold]["recall"].extend(recall.tolist())

        print(f"Processed {min((batch_index + 1) * 4, len(dataset))}/{len(dataset)}")

# Calculate mean scores for each threshold
results = []

for threshold in thresholds:
    results.append({
        "threshold": round(threshold, 2),
        "mean_dice": mean(scores[threshold]["dice"]),
        "mean_precision": mean(scores[threshold]["precision"]),
        "mean_recall": mean(scores[threshold]["recall"]),
    })

# Save results
output = ROOT / "results" / "validation_threshold_sweep.csv"
output.parent.mkdir(exist_ok=True)

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Display comparison
print("\nTHRESHOLD COMPARISON")
print("Threshold | Dice   | Precision | Recall")

for row in results:
    print(
        f"{row['threshold']:.1f}       | "
        f"{row['mean_dice']:.4f} | "
        f"{row['mean_precision']:.4f}    | "
        f"{row['mean_recall']:.4f}"
    )

best = max(results, key=lambda row: row["mean_dice"])

print("\nBEST VALIDATION THRESHOLD")
print(f"Threshold: {best['threshold']:.1f}")
print(f"Mean Dice: {best['mean_dice']:.4f}")
print(f"\nSaved to: {output}")