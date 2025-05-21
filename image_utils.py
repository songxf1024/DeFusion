import os
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from PIL import Image


def sharpen(image, sigma=3):
    '''銳化'''
    # USM锐化增强方法(Unsharpen Mask)
    # 先对原图高斯模糊，用原图减去系数x高斯模糊的图像
    # 再把值Scale到0~255的RGB像素范围
    # 优点：可以去除一些细小细节的干扰和噪声，比卷积更真实
    # （原图像-w*高斯模糊）/（1-w）；w表示权重（0.1~0.9），默认0.6
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(image, (0, 0), sigma)
    usm = cv2.addWeighted(image, 1.5, blur_img, -0.5, 0)
    return usm

def sharpen_laplace(image):
    sharpen_kernal = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    return cv2.filter2D(image, -1, sharpen_kernal)

def claheOfGray(image):
    if image.ndim > 2: image = gray(image)
    # 创建ClAHE对象
    # clipLimit参数表示对比度的大小
    # tileGridSize参数表示每次处理块的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(image)
    return dst

def gray(image):
    # check if we have to convert to gray image
    if image.ndim > 2: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def equalizeHistOfGray(image):
    if image.ndim > 2: image = gray(image)
    return cv2.equalizeHist(image)

def equalizeHistOfColor(image, type='HSV'):
    if image.ndim < 3: return image
    if type == 'HSV':
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])  # V channel
        img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    else:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Y channel
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def claheOfColor(image):
    if image.ndim < 3: return image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if type == 'HSV':
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])  # V channel
        img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    else:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # Y channel
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def gamma(image, gamma=1.0):
    if image.ndim > 2: image = gray(image)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(image.shape)
    for i in range(image.shape[0]): new_imgs[i, 0] = cv2.LUT(np.array(image[i, 0], dtype=np.uint8), table)
    return new_imgs

def preprocess(image):
    ## add your preprocess operations here.

    # image = equalizeHistOfColor(image)
    # accurate：sp=10, sr=10; fast：sp=2, sr=4
    # image = cv2.pyrMeanShiftFiltering(image, sp=2, sr=4)
    #image = sharpen(image)
    image = gray(image)
    # image = clahe(image)
    return image

def largeAndResize(src, scale=2, rect=32):
    n, c, h, w = src.shape
    for i in range(n):
        temp = cv2.resize(src[i].squeeze().numpy(), (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        #temp = sharpen(temp)
        x0, x1 = w * scale // 2 - rect // 2, w * scale // 2 + rect // 2
        y0, y1 = h * scale // 2 - rect // 2, h * scale // 2 + rect // 2
        src[i] = torch.from_numpy(temp[y0:y1, x0:x1]).unsqueeze(dim=0)
    return src

def resizeDatasets(dir_path, ANTIALIAS=True, fast_check=False):
    img_lists = glob(os.path.join(dir_path, '*.JPG'))
    img_lists = img_lists if img_lists else glob(os.path.join(dir_path, '*.jpg'))
    store_path = os.path.join(dir_path, 'resized')
    if fast_check and os.path.exists(store_path):
        return store_path
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    for img in tqdm(img_lists):
        image_path = os.path.join(store_path, img.split(os.sep)[-1])
        if ANTIALIAS:
            image = Image.open(img)
            image_new = image.resize((800, 600), Image.LANCZOS)
            image_new.save(image_path)
        else:
            image = cv2.imread(img)
            image_new = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
            cv2.imwrite(image_path, image_new)
    return store_path

if __name__ == '__main__':
    '''
    img = cv2.imread("/home/sxf/Desktop/pictures/new/a3.jpg")
    shifted = cv2.pyrMeanShiftFiltering(img, 10, 10)
    cv2.imshow("shifted", shifted)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5.0)
    cv2.imshow('MEAN', thresh)
    ret0, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('OTSU', thresh)

    cv2.waitKey()
    exit()
    '''
    img = cv2.imread('/home/sxf/Desktop/my/pictures/new/other/highaffine/resized/ha8.JPG', cv2.IMREAD_COLOR)
    resizeDatasets('/home/sxf/Desktop/net/pictures/new/other/highrotate/scut/')
    exit()
    src = cv2.imread("/home/sxf/Desktop/pictures/new/d11.jpg")
    src = gray(src)
    center_x = 405
    center_y = 285
    rect = 32
    x0, x1 = center_x-rect//2, center_x+rect//2
    y0, y1 = center_y-rect//2, center_y+rect//2
    scale = 2
    src = src[y0:y1, x0:x1]
    src = largeAndResize(src)
    cv2.imshow('res', src)
    '''
    for i in range(2):
        h, w = src.shape
        print(src.shape)
        cv2.imshow('src'+str(i), src)
        src = cv2.resize(src, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('resize'+str(i), src)
        x0, x1 = w * scale // 2 - rect // 2, w * scale // 2 + rect // 2
        y0, y1 = h * scale // 2 - rect // 2, h * scale // 2 + rect // 2
        src = src[y0:y1, x0:x1]
        cv2.imshow('res'+str(i), src)
        '''
    cv2.waitKey(0)
    exit(0)
    cv2.imshow("input", src)
    result = gamma(src)
    cv2.imshow("sharpen_image", result)
    result = sharpen(src, 3)
    cv2.imshow("sharpen_image2", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


