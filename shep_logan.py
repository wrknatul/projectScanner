import cv2
import sys
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

image_size = 150

phantom = 255*shepp_logan_phantom()
photo = resize(phantom, (image_size, image_size), anti_aliasing=True)

if len(sys.argv) != 2:
    print("Usage: python photo_generator.py <output_path>")
    sys.exit(1)

output_path = sys.argv[1]
cv2.imwrite(output_path, photo) 