import numpy as np
import MySQLdb
import pymysql
import math
import  skimage.color as skcolor
import glob
import imageio
import cv2
from xlwt import Workbook
from skimage import img_as_uint
import scipy as sp
from PIL import Image
from skimage.transform import rotate
import skimage as ski
import skimage.io as skio
import matplotlib as mat
import math
from skimage import data, util
import matplotlib.pyplot as plt
import signal
import xlsxwriter

import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
pymysql.install_as_MySQLdb()
def thresholdImage(thresholdvalue,oriImage):
    newImage=oriImage


    for x in range(0,newImage.shape[0]):
        for y in range(0,newImage.shape[1]):
            if(int(newImage[x][y])>thresholdvalue):
                newImage[x][y]=1
            else:
                newImage[x][y]=0


    return newImage
def PadImage(inImg, kernelWidth, kernelHeight):
    if (kernelWidth % 2==1):
        centerX = math.floor(kernelWidth / 2) + 1
        centerY = math.floor(kernelHeight / 2) + 1
    else:

        centerX = kernelWidth
        centerY = 1
    leftExtra = centerX - 1
    rightExtra = kernelWidth - centerX
    topExtra = centerY - 1
    bottomExtra = kernelHeight - centerY

    [r, c] = np.shape(inImg)
    outImg = np.zeros([r + topExtra + bottomExtra, c + leftExtra + rightExtra],dtype=float)

    rowStart = topExtra
    rowEnd = r + topExtra
    colStart = leftExtra
    colEnd = c + leftExtra


    outImg[rowStart: rowEnd, colStart: colEnd] = inImg
    return outImg, rowStart, rowEnd, colStart, colEnd
y=1
row_count=0
workbook = xlsxwriter.Workbook('shaheen.xlsx')
worksheet = workbook.add_worksheet()

for i in range(5):
    x=i+1

    print("img",x)
    img = skio.imread("shaheen_new/shaheen_new_"+str(x)+".jpg", as_grey=True)
# img=PadImage(img,3,3)
    img=img_as_uint(img)

    img=thresholdImage(18000,img)
# img = util.img_as_ubyte(img) > 60


    fig = plt.figure(1)
    fig2 = plt.figure(2)


#     f = fig.add_subplot(241)
#     f.set_title(" sample Image")
# # img=ski.img_as_float(img)
#     f.imshow(img, cmap='gray')
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
    # e = fig.add_subplot(242)
    # e.set_title(" label Image")
    # img=ski.img_as_float(img)
    # e.imshow(limg, cmap='gray')
    objects=ski.measure.regionprops(limg)
    x=0
    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.imshow(limg)
    img_count=y


    file_num=y
    for region in regionprops(limg,img):
    # take regions with large enough areas
        if region.area >= 2236 and region.area<=4000 :
        # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            temp_orientation=region.orientation
            temp_img=limg[ region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]

            temp_img = np.pad(temp_img, (25, 25), 'constant')
            temp_orientation = math.degrees(temp_orientation)
            print("temp orientation", temp_orientation)
            if(temp_orientation>0):

                temp_orientation = 90 - temp_orientation
            else:
                temp_orientation=temp_orientation*-1

            plt.imsave('C:/Users/Faizan/PycharmProjects/iip_project/project/image_shaheen/' + str(img_count) + '.jpg', temp_img,
               cmap=plt.cm.gray)
            temp_img = rotate(temp_img, temp_orientation)

            pixels = (region.major_axis_length + region.minor_axis_length) / 2
            length = region.major_axis_length * 0.15
            width = region.minor_axis_length * 0.15
            plt.imsave('C:/Users/Faizan/PycharmProjects/iip_project/project/images_shaheen/' + str(file_num) + '.jpg', temp_img, cmap=plt.cm.gray)
            worksheet.write(row_count, 0, "image" + str(row_count))
            worksheet.write(row_count, 1, length)
            worksheet.write(row_count, 2, width)
            worksheet.write(row_count, 3, region.mean_intensity)
            worksheet.write(row_count, 4, region.min_intensity)
            worksheet.write(row_count, 5, region.max_intensity)
            img_count = img_count + 1
            file_num=file_num+1
            row_count = row_count + 1

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
    count=0
    print("img_count",img_count)
    f = fig2.add_subplot(241)
    f.set_title(" label Image")

    f.imshow(temp_img, cmap='gray')


# temp_img=PadImage(temp_img,3,3)

    g=fig2.add_subplot(242)


    fig3, ax2 = plt.subplots(figsize=(10, 6))


    path_to_learn_data = glob.glob("images/*.jpg")
    for path in path_to_learn_data:

        im=imageio.imread(path)
        temp_im=im
    # im = img_as_uint(im)
        im = skcolor.rgb2gray(im)



    # im = thresholdImage(127, im)
    #     g.imshow(im, cmap='gray')

        object_img, num_of_objects_1 = ski.measure.label(im, connectivity=im.ndim, return_num=True)
        print("num of objects===",num_of_objects_1)



        # ax2.imshow(object_img)
        for region in regionprops(object_img,im):
            minr, minc, maxr, maxc = region.bbox
            area1=region.area
            print("area of a subimage",area1)
            if(area1>200):
                pixels=(region.major_axis_length+region.minor_axis_length)/2
                length=region.major_axis_length*0.15
                width=region.minor_axis_length*0.15
                print("len",length)
                print("width",width)
                print("average color",region.mean_intensity)
                print("minimum color",region.min_intensity)
                print("maximum color",region.max_intensity)
                # worksheet.write(row_count,0,"image"+str(row_count))
                # worksheet.write(row_count,1,length)
                # worksheet.write(row_count,2,width)
                # worksheet.write(row_count,3,region.mean_intensity)
                # worksheet.write(row_count, 4, region.min_intensity)
                # worksheet.write(row_count, 5, region.max_intensity)
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='white', linewidth=2)
                # row_count=row_count+1

                ax2.add_patch(rect)

    print("row_count",row_count)
# print("temp orientation",temp_orientation)
# temp_orientation=math.degrees(temp_orientation)
# print("temp orientation",temp_orientation)
# temp_orientation=90-temp_orientation
# temp_img=rotate(temp_img,60)
# h=fig2.add_subplot(243)
# h.imshow(temp_img,cmap='gray')

    for x in  range(len(objects)):
        if(objects[x]['area']>=2236 and objects[x]['area']<=4000 ):
            print("area=",objects[x]['area'])
            print("major axis length=",objects[x]['major_axis_length'])
            print("minor axis length=", objects[x]['minor_axis_length'])
        # print("temp orientation", temp_orientation)
        # temp_img = np.pad(temp_img, (25, 25), 'constant')
        # temp_orientation = math.degrees(temp_orientation)
        # print("temp orientation", temp_orientation)
        # temp_orientation = 90 - temp_orientation
        # temp_img = rotate(temp_img, temp_orientation)
        # h = fig2.add_subplot(img_count)
        # h.imshow(temp_img, cmap='gray')

            count=count+1

    ax.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    print("num of objects in image",num_of_objects)
    print("num of objects in image",count)
    y=y+10

workbook.close()
    # plt.show()