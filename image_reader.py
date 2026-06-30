import cv2
import numpy as np

image = cv2.imread("image.jpeg")



print("shape:", image.shape)
print("Data type:",image.dtype)
print("First pixel:", image[0,0])

cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()