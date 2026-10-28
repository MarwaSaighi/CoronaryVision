import numpy as np
import torch
from PIL import Image

from src.inference import segment_image


class FakeModel(torch.nn.Module):
    def forward(self, x):
        # Predict background everywhere
        return torch.full(
            (x.shape[0], 1, x.shape[2], x.shape[3]),
            -2.0
        )


def test_segment_image():
    # Create a small artificial RGB image
    image = Image.new("RGB", (64, 48), color="gray")

    model = FakeModel()

    original, overlay, mask, probabilities = segment_image(
        image, model
    )

    # Verify image dimensions
    assert original.size == (512, 512)
    assert overlay.size == (512, 512)

    # Verify prediction dimensions
    assert mask.shape == (512, 512)
    assert probabilities.shape == (512, 512)

    # The fake model predicts no vessels
    assert not np.any(mask)

    # Every probability must be at least 0
    assert np.all(probabilities >= 0)

    # Every probability must be at most 1
    assert np.all(probabilities <= 1)