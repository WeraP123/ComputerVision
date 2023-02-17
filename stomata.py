# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:14:30 2023

@author: bb18115
"""
import sxcv
import cv2

FILENAME = 'Images/stomata/005-ab'

def get_centres(filename):
    f = open(FILENAME+'.gt', 'r')
    points = []
    f.readline()
    for line in f.readlines():
        a = [int(word) for word in line.strip(' ').split(' ') if word not in (' ','')]
        points.append(tuple(a))
    return points
        
def load_img(filename):
    image = cv2.imread(FILENAME+'.jpg')
    g_im = cv2.imread(FILENAME+'.jpg',cv2.IMREAD_GRAYSCALE)
    neg_im = 255-g_im
    return image, g_im, neg_im
    
def align_centres_to_img(img,centres):
    if len(img.shape) == 2 or (len(img.shape)== 3 and img.shape[-1]==1):
        img = cv2.merge([img]*3)
    point_image = img.copy()
    for coor in centres:
        cv2.circle(point_image,(coor[1],coor[0]),radius=1, color=(255,0,0),thickness=1)
    return point_image
    
def sobel(img):
    sobel_x = cv2.filter2D(src=img, ddepth=-1, kernel=sxcv.create_mask('sobelx'))
    sobel_y = cv2.filter2D(src=img, ddepth=-1, kernel=sxcv.create_mask('sobely'))
    return sobel_x + sobel_y
    
def sobel_root_squared(img):
    sobel_x = cv2.filter2D(src=img, ddepth=-1, kernel=sxcv.create_mask('sobelx'))
    sobel_y = cv2.filter2D(src=img, ddepth=-1, kernel=sxcv.create_mask('sobely'))
    return ((sobel_x**2+sobel_y**2)**0.5).clip(0,255).astype('uint8')
def laplacian_of_gaussian(img):
    gaussian = cv2.filter2D(src=img, ddepth=-1, kernel=sxcv.create_mask('gaussian5'))
    return cv2.filter2D(src=gaussian, ddepth=-1, kernel=sxcv.create_mask('laplacian'))
    
def show_images(images_dict):
    for name,image in images_dict.items():
        cv2.imshow(name,image)
    cv2.waitKey(0)
def linear_operator(im,a,b):
    return ((g_im*a)+b).clip(0,255).astype('uint8')
centres = get_centres(FILENAME)
images_dict = {}
im, g_im, neg_im = load_img(FILENAME)

mult_im = linear_operator(g_im,1.9,-127)
gaussian = cv2.filter2D(src=mult_im, ddepth=-1, kernel=sxcv.create_mask('gaussian5'))
laplacian_of_gaussian = laplacian_of_gaussian(mult_im)
sobel = sobel(gaussian)
sobel_rs = sobel_root_squared(gaussian)

formatted_image = (g_im+sobel).clip(0,255)
_,bin_im = cv2.threshold(formatted_image,0,255,cv2.THRESH_OTSU)
points = align_centres_to_img(im,centres)
bin_points = align_centres_to_img(bin_im,centres)
images_dict = {
    'original':im,
    'grayscale':g_im,
    'multiplied':mult_im,
    'gaussian': gaussian,
    'LoG': laplacian_of_gaussian+100,
    'g+sobel': sobel+100,
    'g+sobel_rs':sobel_rs+100,
    'formatted_image':formatted_image,
    'binary':bin_im,
    'centres': points,
    'bin_centres': bin_points
}
show_images(images_dict)
