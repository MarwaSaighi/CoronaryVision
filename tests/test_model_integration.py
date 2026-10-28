import numpy as np
import pytest
from PIL import Image

from src.inference import (
    CHECKPOINT_PATH,
    load_model,
    segment_image,
)


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Trained checkpoint not available locally"
)
def test_real_model_inference():

    # Create an artificial grayscale image
    image = Image.new("L", (512, 512), color=100)

    # Load our REAL trained U-Net
    model = load_model()

    # Run our REAL inference pipeline
    original, overlay, mask, probabilities = segment_image(
        image, model
    )

    # Verify output dimensions
    assert original.size == (512, 512)
    assert overlay.size == (512, 512)
    assert mask.shape == (512, 512)
    assert probabilities.shape == (512, 512)

    # Verify probabilities are valid numbers
    assert np.all(np.isfinite(probabilities))

    # Verify the binary mask matches our threshold
    assert np.array_equal(mask, probabilities >= 0.5)