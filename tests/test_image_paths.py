from src.image_paths import image_exists

def test_existing_file(tmp_path):
    test_file = tmp_path / "sample.png"
    test_file.touch()  # Create an empty file
    assert image_exists(tmp_path, "sample.png")

def test_missing_file(tmp_path):
    assert not image_exists(tmp_path, "nonexistent.png")