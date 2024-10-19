import cv2
import numpy as np
photo = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        if (i - 50) ** 2 + (j - 50) ** 2 <= 26**2:
            photo[i][j] = 255
cv2.imwrite("kaggle/working/projectScanner/data/goal_photo.jpg", photo) 