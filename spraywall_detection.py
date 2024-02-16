import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


#Reading and conversions
baseImage_bgr = cv.imread('sprywallclean.png')
baseImage_rbg = cv.cvtColor(baseImage_bgr,cv.COLOR_BGR2RGB)
img_grey = cv.imread('sprywallclean.png',cv.IMREAD_GRAYSCALE)
img_hsv = cv.cvtColor(baseImage_bgr, cv.COLOR_BGR2HSV)
h,s,v = cv.split(img_hsv)





#filtering
    #h
bilateral_filtered_h = cv.bilateralFilter(h ,9 ,75,75)
gaussian_filtered_h = cv.GaussianBlur(h ,(5,5) ,0 )
    #s
bilateral_filtered_s =cv.bilateralFilter(s,9 ,75 , 75)
gaussian_filtered_s = cv.GaussianBlur(s , (5,5),0)



#Thresholding
    #h
otsureth,otsu_thresh_h=cv.threshold( bilateral_filtered_h, 0 ,255 , cv.THRESH_BINARY+cv.THRESH_OTSU)
    #s
otsureth,otsu_thresh_s=cv.threshold( bilateral_filtered_s, 0 ,255 , cv.THRESH_BINARY+cv.THRESH_OTSU)

reth,thresh_h =cv.threshold(h,127,150,cv.THRESH_BINARY)
rets,thresh_s =cv.threshold(s,200,255,cv.THRESH_BINARY)
retv,thresh_v = cv.threshold(v , 127,150,cv.THRESH_BINARY)
ret,threshgrey =cv.threshold(img_grey,127, 150,cv.THRESH_BINARY)
#morphology
kernel = np.ones((3,3),np.uint8)
closing_grey = cv.morphologyEx(threshgrey, cv.MORPH_CLOSE, kernel)
closing_h = cv.morphologyEx(thresh_h, cv.MORPH_CLOSE, kernel)
closing_s = cv.morphologyEx(thresh_s, cv.MORPH_CLOSE, kernel)
closing_v = cv.morphologyEx(thresh_v, cv.MORPH_CLOSE, kernel)

opened_s =cv.morphologyEx(closing_s,cv.MORPH_OPEN ,kernel)
#canny
canny_s=cv.Canny(bilateral_filtered_s,50,200)

contours,_= cv.findContours(canny_s, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

cv.drawContours(canny_s,contours , -1 ,(255,0,0) ,2)


#maths
inter_hs = cv.bitwise_or(thresh_h ,thresh_s)






fig = plt.figure(figsize=(10, 13)) 
rows = 3
columns = 3

fig.add_subplot(rows,columns ,1)
plt.title('h')
plt.imshow(h)

fig.add_subplot(rows,columns , 2)
plt.title('h bil filter')
plt.imshow(bilateral_filtered_h)

fig.add_subplot(rows,columns , 3)
plt.title('otsu tresh h after bil filter')
plt.imshow(otsu_thresh_h)

fig.add_subplot(rows,columns ,4)
plt.title('s')
plt.imshow(s)

fig.add_subplot(rows,columns , 5)
plt.title('s bil filter')
plt.imshow(bilateral_filtered_s)

fig.add_subplot(rows,columns , 6)
plt.title('otsu tresh s after bil filter')
plt.imshow(otsu_thresh_s)

fig.add_subplot(rows,columns , 7)
plt.title('canny detection and contours')
plt.imshow(canny_s)







plt.tight_layout()
plt.show()

