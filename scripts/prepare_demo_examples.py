import csv
import shutil
from pathlib import Path
from statistics import median

import kagglehub

ROOT = Path(__file__).resolve().parents[1]

# Read our existing validation evaluation results
metrics_path = ROOT / "results" / "validation_image_metrics.csv"

with open(metrics_path, newline="", encoding="utf-8") as file:
    results = list(csv.DictReader(file))

# Find an image whose Dice is close to the median
median_dice = median(float(row["dice"]) for row in results)

typical = min(
    (
        row for row in results
        if row["filename"] not in {"89.png", "109.png"}
    ),
    key=lambda row: abs(float(row["dice"]) - median_dice),
)

# Locate the public ARCADE validation images
dataset_root = Path(
    kagglehub.dataset_download("nirmalgaud/arcade-dataset")
)

source_dir = dataset_root / "arcade" / "syntax" / "val" / "images"

# Copy only three images into our application
output_dir = ROOT / "assets" / "examples"
output_dir.mkdir(parents=True, exist_ok=True)

sample_names = [
    typical["filename"],
    "89.png",
    "109.png",
]

for filename in sample_names:
    shutil.copy2(source_dir / filename, output_dir / filename)
    print("Added:", filename)

print("Example images are ready!")