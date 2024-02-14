import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
img = cv.imread('sprywallclean.png',cv.IMREAD_GRAYSCALE)
ret,thresh =cv.threshold(img,127, 150,cv.THRESH_BINARY)
plt.imshow(thresh,'gray',vmin=0,vmax=255)
plt.xticks([]),plt.yticks([])



plt.show()
