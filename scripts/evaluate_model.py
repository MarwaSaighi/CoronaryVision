import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import kagglehub
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from src.arcade_dataset import ArcadeSegmentationDataset


# 1. Read the requested dataset split
parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    choices=["val", "test"],
    required=True
)

args = parser.parse_args()

split = args.split
split_name = "validation" if split == "val" else "test"


# 2. Locate the dataset and checkpoint
ROOT = Path(__file__).resolve().parents[1]

dataset_root = Path(
    kagglehub.dataset_download("nirmalgaud/arcade-dataset")
)

data_dir = dataset_root / "arcade" / "syntax" / split

dataset = ArcadeSegmentationDataset(
    images_dir=data_dir / "images",
    annotation_file=data_dir / "annotations" / f"{split}.json"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# 3. Load the selected model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

checkpoint = torch.load(
    ROOT / "models" / "unet_imagenet_best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# 4. Evaluate each image
results = []

with torch.inference_mode():
    for batch_index, (images, masks) in enumerate(loader):

        probabilities = torch.sigmoid(model(images))

        predictions = probabilities >= 0.5
        targets = masks >= 0.5

        for j in range(images.shape[0]):

            pred = predictions[j]
            gt = targets[j]

            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())

            dice_den = 2 * tp + fp + fn
            iou_den = tp + fp + fn

            dice = 2 * tp / dice_den if dice_den else 1.0
            iou = tp / iou_den if iou_den else 1.0

            precision = (
                tp / (tp + fp)
                if tp + fp else (1.0 if tp + fn == 0 else 0.0)
            )

            recall = tp / (tp + fn) if tp + fn else 1.0

            index = batch_index * loader.batch_size + j

            results.append({
                "filename": dataset.images[index]["file_name"],
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn
            })

        print(f"Processed {len(results)}/{len(dataset)} images")


# 5. Calculate dataset-level metrics
summary = {
    "model": "unet_imagenet_best",
    "split": split_name,
    "threshold": 0.5,
    "num_images": len(results),
    "mean_dice": mean(r["dice"] for r in results),
    "mean_iou": mean(r["iou"] for r in results),
    "mean_precision": mean(r["precision"] for r in results),
    "mean_recall": mean(r["recall"] for r in results)
}


# 6. Save results
results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / f"{split_name}_image_metrics.csv"
json_path = results_dir / f"{split_name}_summary.json"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)


# 7. Print results
print(f"\n{split_name.upper()} RESULTS")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nResults saved successfully.")