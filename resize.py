import numpy as np
import cv2
import os

path = "./output_p/leaf_mask/growliflower/test2"


img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
print(np.unique(img))
cv2.imwrite("./output.png",img)