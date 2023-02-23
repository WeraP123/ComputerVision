import cv2
import numpy
import sxcv

im = cv2.imread('Images/dice.jpg')
im = cv2.resize(im,(640,480))
hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
v, h = sxcv.histogram(gray_im)
sxcv.plot_histogram(v,h,'gray')
bim = sxcv.binarize(gray_im,195)
contours, hierarchy = cv2.findContours(bim, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
display = cv2.drawContours(im,contours, -1,(0,0,255),1)
sxcv.display(bim)
sxcv.display(display)

dice = []    # list of dice contours
spots = []   # list of spot contours

# Find dice contours.
for (i, c) in enumerate(hierarchy[0]):
    if c[3] == -1:
        dice.append (i)

# Find spot contours.
for (i, c) in enumerate(hierarchy[0]):
    if c[3] in dice:
        spots.append (i)

print ("Dice roll total:", len (spots)) 