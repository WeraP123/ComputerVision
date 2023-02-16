# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:14:30 2023

@author: bb18115
"""
import sxcv
import cv2
FILENAME = 'Images/stomata/005-ab'
f = open(FILENAME+'.gt', 'r')
points = []
f.readline()
for line in f.readlines():
    a = [int(word) for word in line.strip(' ').split(' ') if word not in (' ','')]
    points.append(tuple(a))

image = cv2.imread(FILENAME+'.jpg')
mult_im = cv2.multiply(image,(1),scale=1.2).astype('uint8')
#_,bin_im = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=sxcv.create_mask('blur5'))
formatted_image = image-filtered_image
point_image = image.copy()
for coor in points:
    cv2.circle(point_image,(coor[1],coor[0]),radius=1, color=(255,0,0),thickness=1)
cv2.imshow('image',image)
cv2.imshow('point_image',point_image)
cv2.imshow('multiple image',mult_im)
#cv2.imshow('bin_im',bin_im)
cv2.imshow('filtered',filtered_image)
cv2.imshow('formatted_image',formatted_image)
cv2.waitKey(0)
