import numpy as np
from skimage import img_as_uint
import scipy as sp
import skimage as ski
import skimage.io as skio
import matplotlib as mat
import math
from skimage.transform import rotate
from skimage import data, util
import matplotlib.pyplot as plt
import signal
import cv2
from PIL import Image

import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def thresholdImage(thresholdvalue,oriImage):
    newImage=oriImage


    for x in range(0,newImage.shape[0]):
        for y in range(0,newImage.shape[1]):
            if(int(newImage[x][y])>thresholdvalue):
                newImage[x][y]=1
            else:
                newImage[x][y]=0


    return newImage
img = skio.imread("1.jpg", as_grey=True)
img=img_as_uint(img)
print(img)
img=thresholdImage(12000,img)
# img = util.img_as_ubyte(img) > 60
print( img)

fig = plt.figure(1)

f = fig.add_subplot(241)
f.set_title(" sample Image")
# img=ski.img_as_float(img)
f.imshow(img, cmap='gray')
# apply threshold
# thresh = threshold_otsu(img)
# bw = closing(img > thresh, square(3))
#
# # remove artifacts connected to image border
# cleared = clear_border(bw)
#
# # label image regions
# label_image,num_of_objects = label(cleared,return_num=True)
# image_label_overlay = label2rgb(label_image, image=img)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(image_label_overlay)
#
# for region in regionprops(label_image):
#     # take regions with large enough areas
#     if region.area >= 950:
#         # draw rectangle around segmented coins
#         minr, minc, maxr, maxc = region.bbox
#         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                   fill=False, edgecolor='red', linewidth=2)
#         ax.add_patch(rect)
# for region in regionprops(label_image):
#     if region.area>=950:
#         print(region.area)
#         print("major axis length=",region.major_axis_length)
#         print("minor axis length=", region.minor_axis_length)
#
# ax.set_axis_off()
# plt.tight_layout()
limg,num_of_objects=ski.measure.label(img,connectivity=img.ndim,return_num=True)
e = fig.add_subplot(242)
e.set_title(" label Image")
img=ski.img_as_float(img)
e.imshow(limg, cmap='gray')
objects=ski.measure.regionprops(limg)
x=0
fig, ax = plt.subplots(figsize=(10, 6))
fig, ax2 = plt.subplots(figsize=(10, 6))

ax.imshow(limg)
ax2.imshow(limg)
for region in regionprops(limg,img):
    # take regions with large enough areas
    if region.area >= 700 and region.area<=1400 :
        # draw rectangle around segmented coins

        minr, minc, maxr, maxc = region.bbox
        # img = limg
        # area = (maxc,minc,maxr,minr)
        # cropped_img = img[minc:maxc,minr:maxr]


        intensity_img=region.image
        # img2=Image.open(intensity_img,'r')
        # img2.rotate(45)
        # intensity_img=img2
        cords=region.coords

        img1=limg[cords]
        print( "img1",img1)
        # cv2.imshow('img',img1)
        print(region.extent)
        # img2= cv2.rotate(int(intensity_img),90)
        #
        f = fig.add_subplot(243)
        f.imshow(intensity_img, cmap='gray')
        print(cords)


        print(region.bbox)



        # region.orientation=1.4545131631837391
        if(region.orientation==1.4545131631837391):
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='white', linewidth=2)
        else:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # ax2.add_patch(region)
count=0
for x in  range(len(objects)):
    if(objects[x]['area']>=700 and objects[x]['area']<=1400 ):
        print("area=",objects[x]['area'])
        print("major axis length=",objects[x]['major_axis_length'])
        print("minor axis length=", objects[x]['minor_axis_length'])
        print("angle",objects[x]['orientation'])
        count = count + 1



        # print("area=", objects[x]['area'])


ax.set_axis_off()
plt.tight_layout()
print("num of objects in image",num_of_objects)
print("num of objects in image",count)
ax2.set_axis_off()
plt.tight_layout()
plt.show()