import os

def build_image_path(folder, filename):
    return os.path.join(folder, filename)

result = build_image_path("data/images", "922.png")
print(result)

def image_exists(folder, filename):
    return os.path.isfile(build_image_path(folder, filename))
