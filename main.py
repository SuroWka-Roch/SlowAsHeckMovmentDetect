#!/usr/bin/env python3.7
import cv2 as cv
import numpy as np

tło = cv.imread("./fig/ruch0.jpg")
fig = cv.imread("./fig/ruch1.jpg")

hight = tło.shape[0]
width = tło.shape[1]

diff = np.zeros((hight, width, 3), np.uint8)
diff2 = np.zeros((hight, width, 3), np.uint8)

for i in range(hight):
    for j in range(width):
        for a in range(3):
            diff[i][j][a] = 0 if tło[i][j][a] < fig[i][j][a] else tło[i][j][a] - fig[i][j][a]
            diff2[i][j][a] = 0 if fig[i][j][a] < tło[i][j][a] else fig[i][j][a] - tło[i][j][a]

diff = ~diff
diff2 = ~diff2

diff = diff+diff2

img_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
t, img_bin = cv.threshold(img_gray, 240, 255, cv.THRESH_BINARY )

im_flood = ~img_bin.copy()

mask = np.zeros((hight+2, width+2), np.uint8)
cv.floodFill(im_flood, mask, (0,0), 255)

pic_final = im_flood & img_bin

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
pic_final = cv.dilate(pic_final,kernel,iterations=2)

cv.imshow("name", img_bin)

cv.imshow("name2", pic_final)

contours,hier = cv.findContours(pic_final, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(fig, contours, -1, (145,144,30), 2)

cv.imshow("name2", fig)

cv.waitKey()
