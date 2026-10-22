from pathlib import Path

import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import json
from src.mask_utils import create_mask_from_annotations


# Load the trained model once
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "models"
        / "unet_imagenet_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def segment_image(uploaded_image, model):
    # Preprocessing
    image = uploaded_image.convert("L")
    image = image.resize((512, 512), Image.Resampling.BILINEAR)

    image_array = np.array(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(image_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Model inference
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)

    probability_map = probabilities[0, 0].cpu().numpy()
    mask = probability_map >= 0.5

    # Create red segmentation overlay
    overlay = np.array(image.convert("RGB")).copy()

    overlay[mask] = (
        0.6 * overlay[mask]
        + 0.4 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return image, Image.fromarray(overlay), mask, probability_map


# Web interface
st.title("CoronaryVision")

st.write(
    "Coronary artery segmentation using a U-Net "
    "with a ResNet-18 encoder."
)

st.warning("Research demonstration only. Not for clinical use.")

uploaded_file = st.file_uploader(
    "Upload a coronary angiography image",
    type=["png", "jpg", "jpeg"]
)

annotation_file = st.file_uploader(
    "Upload ARCADE ground-truth annotations (test.json)",
    type=["json"]
)

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Generating segmentation..."):
        original, overlay, predicted_mask, probability_map = segment_image(image, model)

    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original angiography", width="stretch")

    with col2:
        st.image(overlay, caption="Predicted artery segmentation", width="stretch")
    # Display probability map before thresholding
    st.subheader("Model probability map")

    probability_image = Image.fromarray(
        (probability_map * 255).astype(np.uint8)
    )

    st.image(
        probability_image,
        caption="White = high vessel probability; black = low vessel probability",
        width="stretch"
    )
    if annotation_file is not None:
        coco = json.loads(annotation_file.getvalue().decode("utf-8"))

        matches = [
            info for info in coco["images"]
            if Path(info["file_name"]).name == uploaded_file.name
        ]

        if len(matches) != 1:
            st.error("Image not found in this annotation file.")
        else:
            image_id = matches[0]["id"]

            annotations = [
                ann for ann in coco["annotations"]
                if ann["image_id"] == image_id
            ]

            gt_image = create_mask_from_annotations(
                original.size, annotations
            )

            gt_mask = np.array(gt_image, dtype=bool)
            gt_overlay = np.array(original.convert("RGB")).copy()

            # Show ground truth in green
            gt_overlay[gt_mask] = (
                0.6 * gt_overlay[gt_mask]
                + 0.4 * np.array([0, 255, 0])
            ).astype(np.uint8)

            st.image(
                Image.fromarray(gt_overlay),
                caption="Ground truth — annotated artery regions (green)"
            )
            # Compare prediction with ground truth
            tp = predicted_mask & gt_mask
            fp = predicted_mask & ~gt_mask
            fn = ~predicted_mask & gt_mask

            # Count pixels
            tp_count = np.count_nonzero(tp)
            fp_count = np.count_nonzero(fp)
            fn_count = np.count_nonzero(fn)

            # Calculate Dice and IoU
            dice_denominator = 2 * tp_count + fp_count + fn_count
            iou_denominator = tp_count + fp_count + fn_count

            dice = (
                2 * tp_count / dice_denominator
                if dice_denominator > 0 else 1.0
            )

            iou = (
                tp_count / iou_denominator
                if iou_denominator > 0 else 1.0
            )

            # Display metrics
            st.subheader("Segmentation evaluation")

            metric1, metric2 = st.columns(2)

            metric1.metric("Dice score", f"{dice:.4f}")
            metric2.metric("IoU score", f"{iou:.4f}")

            # Create color-coded error map
            error_map = np.array(original.convert("RGB")).copy()

            regions = [
                (tp, [0, 255, 0]),    # Green: correct vessel
                (fp, [255, 0, 0]),    # Red: false positive
                (fn, [0, 100, 255])   # Blue: missed vessel
            ]

            for region, color in regions:
                error_map[region] = (
                    0.5 * error_map[region]
                    + 0.5 * np.array(color)
                ).astype(np.uint8)

            st.image(
                Image.fromarray(error_map),
                caption="Error map: green = TP, red = FP, blue = FN"
            )

            st.caption(
                f"Correct vessel pixels: {tp_count:,} | "
                f"False positives: {fp_count:,} | "
                f"Missed vessel pixels: {fn_count:,}"
            )
