from PIL import Image
from src.image_loading import load_grayscale_image

def test_load_grayscale_image(tmp_path):
    image_path = tmp_path / "sample.png"

    original = Image.new("RGB", (8, 8), color="red")
    original.save(image_path)

    grayscale = load_grayscale_image(image_path)

    assert grayscale.mode == "L"
    assert grayscale.size == (8, 8)

import pytest

def test_missing_image(tmp_path):
    missing_path = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        load_grayscale_image(missing_path)