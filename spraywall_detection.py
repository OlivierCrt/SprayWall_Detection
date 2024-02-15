import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np



baseImage = cv.imread('sprywallclean.png',1)
img_grey = cv.imread('sprywallclean.png',cv.IMREAD_GRAYSCALE)
img_hsv = cv.cvtColor(baseImage, cv.COLOR_BGR2HSV)

h,s,v = cv.split(img_hsv)
#Thresholding
reth,thresh_h =cv.threshold(h,127,150,cv.THRESH_BINARY)
rets,thresh_s =cv.threshold(s,127,150,cv.THRESH_BINARY)
retv,thresh_v = cv.threshold(v , 127,150,cv.THRESH_BINARY)
ret,threshgrey =cv.threshold(img_grey,127, 150,cv.THRESH_BINARY)
#morphology
kernel = np.ones((2,2),np.uint8)
closing_grey = cv.morphologyEx(threshgrey, cv.MORPH_CLOSE, kernel)
closing_h = cv.morphologyEx(thresh_h, cv.MORPH_CLOSE, kernel)
closing_s = cv.morphologyEx(thresh_s, cv.MORPH_CLOSE, kernel)
closing_v = cv.morphologyEx(thresh_v, cv.MORPH_CLOSE, kernel)

opened_s =cv.morphologyEx(closing_s,cv.MORPH_OPEN ,kernel)






fig = plt.figure(figsize=(10, 13)) 
rows = 4
columns = 4
fig.add_subplot(rows, columns, 1) 
plt.title('Base image')
plt.imshow(cv.cvtColor(baseImage,cv.COLOR_BGR2RGB))
fig.add_subplot(rows, columns, 2) 
plt.title('Greyscale image')
plt.imshow(cv.cvtColor(img_grey,cv.COLOR_GRAY2RGB))
fig.add_subplot(rows,columns,3)



plt.title('Threshold')
plt.imshow(threshgrey,'gray',vmin=0,vmax=255)
fig.add_subplot(rows, columns, 4)
plt.title('Morpho closing')
plt.imshow(closing_grey,'gray',vmin=0,vmax=255)
fig.add_subplot(rows, columns ,5)
plt.title('H')
plt.imshow(thresh_h)

fig.add_subplot(rows, columns ,6)
plt.title('s')
plt.imshow(thresh_s)

fig.add_subplot(rows, columns ,7)
plt.title('v')
plt.imshow(thresh_v)

fig.add_subplot(rows, columns ,8)
plt.title('h_closed')
plt.imshow(closing_h)

fig.add_subplot(rows, columns ,9)
plt.title('s_closed')
plt.imshow(closing_s)

fig.add_subplot(rows, columns ,10)
plt.title('v_closed')
plt.imshow(closing_v)

fig.add_subplot(rows, columns ,11)
plt.title('s opened')
plt.imshow(opened_s)

plt.tight_layout()



plt.show()
