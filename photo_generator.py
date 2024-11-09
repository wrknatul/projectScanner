import cv2
import numpy as np
import sys

photo = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if (i - 50) ** 2 + (j - 50) ** 2 <= 26**2:
            photo[i][j] = 255

if len(sys.argv) != 2:
    print("Usage: python photo_generator.py <output_path>")
    sys.exit(1)

output_path = sys.argv[1]
cv2.imwrite(output_path, photo) 