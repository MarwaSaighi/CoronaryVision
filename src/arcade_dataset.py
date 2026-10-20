from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.image_loading import load_grayscale_image
from src.mask_utils import create_mask_from_annotations


class ArcadeSegmentationDataset(Dataset):

    def __init__(self, images_dir, annotation_file):
        self.images_dir = Path(images_dir)

        with open(annotation_file, encoding="utf-8") as f:
            coco = json.load(f)

        self.images = coco["images"]

        self.annotations_by_image = defaultdict(list)

        for annotation in coco["annotations"]:
            image_id = annotation["image_id"]
            self.annotations_by_image[image_id].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]

        image_path = self.images_dir / image_info["file_name"]

        image = load_grayscale_image(image_path)

        annotations = self.annotations_by_image[image_info["id"]]

        mask = create_mask_from_annotations(
            image.size, annotations
        )

        image_array = np.array(image, dtype=np.float32) / 255.0
        mask_array = np.array(mask, dtype=np.float32)

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return image_tensor, mask_tensor