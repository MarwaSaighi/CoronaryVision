from src.mask_utils import create_empty_mask, draw_polygon, create_mask_from_annotations


def test_create_empty_mask():
    width, height = 6, 4
    mask = create_empty_mask(width, height)

    assert mask.size == (width, height)
    assert mask.mode == "L"
    assert mask.getpixel((0, 0)) == 0


def test_draw_polygon():
    width, height = 6, 6
    mask = create_empty_mask(width, height)

    points = [(1, 1), (4, 1), (4, 4), (1, 4)]
    mask_with_polygon = draw_polygon(mask, points)

    assert mask_with_polygon.getpixel((2, 2)) == 1
    assert mask_with_polygon.getpixel((0, 0)) == 0


def test_create_mask_from_annotations():
    annotations = [
        {
            "category_id": 1,
            "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]],
        },
        {
            "category_id": 26,
            "segmentation": [[5, 5, 7, 5, 7, 7, 5, 7]],
        },
    ]

    mask = create_mask_from_annotations((8, 8), annotations)

    assert mask.size == (8, 8)
    assert mask.getpixel((2, 2)) == 1
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((6, 6)) == 0