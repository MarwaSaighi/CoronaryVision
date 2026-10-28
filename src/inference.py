from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
from huggingface_hub import hf_hub_download


# Find the project root
ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "models" / "unet_imagenet_best.pt"

def get_checkpoint_path():
    """Use the local checkpoint or download it from Hugging Face."""

    if CHECKPOINT_PATH.is_file():
        return CHECKPOINT_PATH

    return Path(
        hf_hub_download(
            repo_id="saighim/CoronaryVision-UNet",
            filename="unet_imagenet_best.pt",
            revision="b917627b4a1ba4fdd82201892c3fae63df98d39e",
        )
    )

def load_model():
    """Load the trained U-Net on CPU."""

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )

    checkpoint = torch.load(
        get_checkpoint_path(),
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def segment_image(image, model, threshold=0.5):
    """Predict coronary artery regions in an uploaded image."""

    # Preprocessing
    image = image.convert("L")
    image = image.resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(image_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Inference
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    probability_map = probabilities[0, 0].cpu().numpy()

    # Binary segmentation
    mask = probability_map >= threshold

    # Red overlay
    overlay = np.array(image.convert("RGB")).copy()

    overlay[mask] = (
        0.6 * overlay[mask]
        + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return image, Image.fromarray(overlay), mask, probability_map