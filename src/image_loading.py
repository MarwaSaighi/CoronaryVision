from PIL import Image

def load_grayscale_image(image_path):
    with Image.open(image_path) as image:
        grayscale_image = image.convert("L")
    return grayscale_image