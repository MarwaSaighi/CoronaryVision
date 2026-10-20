from PIL import Image

def create_empty_mask(width, height):
    """
    Create an empty mask image with the specified width and height.

    Args:
        width (int): The width of the mask.
        height (int): The height of the mask.

    Returns:
        PIL.Image: An empty mask image.
    """
    return Image.new("L", (width, height), 0)

def draw_polygon(mask, points):
    """
    Draw a polygon on the given mask image.

    Args:
        mask (PIL.Image): The mask image to draw on.
        points (list): A list of (x, y) tuples representing the polygon vertices.

    Returns:
        PIL.Image: The mask image with the polygon drawn.
    """
    from PIL import ImageDraw

    drawer = ImageDraw.Draw(mask)
    drawer.polygon(points, fill=1)
    return mask

def create_mask_from_annotations(image_size, annotations):
    mask = create_empty_mask(*image_size)

    for annotation in annotations:
        # Exclude stenosis annotations for now
        if annotation["category_id"] == 26:
            continue

        for polygon in annotation["segmentation"]:
            if len(polygon) < 6 or len(polygon) % 2 != 0:
                raise ValueError("Invalid polygon coordinates")

            points = list(zip(polygon[::2], polygon[1::2]))
            mask = draw_polygon(mask, points)

    return mask
