# -*- coding: utf-8 -*-
# @Time    : 2022/1/20 下午1:56
# @Author  : 小锋学长生活大爆炸
# @FileName: segmentation.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/sxf1061700625
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as pylab
import numpy as np
from scipy import ndimage as ndi
from skimage.color import label2rgb

from image_utils import equalizeHistOfColor

#matplotlib.use('TkAgg')

matplotlib.rcParams['font.sans-serif'] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 分水岭算法实现图像自动分割的步骤：
# 1: 对原图像进行灰度化处理，并进行边缘检测或二值化
# 距離變換?
# 2: 查找轮廓，并且把轮廓信息按不同的编号绘制在标记图像上，即标记种子点，将其传给marker参数
# 3: 进行分水岭算法检测
# 4: 绘制分割出来的区域，使用颜色进行填充

class Segment:
    def __init__(self):
        pass

    def sobel_sxf(self, gray, size):
        # ksize是指核的大小,只能取奇数，影响边缘的粗细
        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=size)
        y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=size)

        # 转回uint8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    def drawColor(self, marks, img):
        # 生成随机颜色
        colorTab = np.zeros((np.max(marks) + 1, 3))
        # 生成0~255之间的随机数
        for i in range(len(colorTab)):
            aa = np.random.uniform(0, 255)
            bb = np.random.uniform(0, 255)
            cc = np.random.uniform(0, 255)
            colorTab[i] = np.array([aa, bb, cc], np.uint8)

        bgrImage = np.zeros(img.shape, np.uint8)
        # 遍历marks每一个元素值，对每一个区域进行颜色填充
        for i in range(marks.shape[0]):
            for j in range(marks.shape[1]):
                # index值一样的像素表示在一个区域
                index = marks[i][j]
                # 判断是不是区域与区域之间的分界,如果是边界(-1)，则使用白色显示
                if index == -1:
                    bgrImage[i][j] = np.array([255, 255, 255])
                else:
                    bgrImage[i][j] = colorTab[index]
        cv2.imshow('After ColorFill', bgrImage)

        # 填充后与原始图像融合
        # 实现以不同的权重将两幅图片叠加，对于不同的权重，叠加后的图像会有不同的透明度
        # alpha(0.55)是第一幅图片中元素的权重，beta(0.45)是第二个的权重， gamma(0)是加到最后结果上的一个值
        result = cv2.addWeighted(img, 0.55, bgrImage, 0.45, 0)
        cv2.imshow('addWeighted', result)
        # cv2.waitKey(5000)

    def checkOverlap(self, boxa, boxb):
        x1, y1, w1, h1 = boxa
        x2, y2, w2, h2 = boxb
        if (x1 > x2 + w2):
            return 0
        if (y1 > y2 + h2):
            return 0
        if (x1 + w1 < x2):
            return 0
        if (y1 + h1 < y2):
            return 0
        colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
        rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = colInt * rowInt
        area1 = w1 * h1
        area2 = w2 * h2
        return overlap_area / (area1 + area2 - overlap_area)

    def unionBox(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0] + a[2], b[0] + b[2]) - x
        h = max(a[1] + a[3], b[1] + b[3]) - y
        return [x, y, w, h]

    def intersectionBox(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        if w < 0 or h < 0:
            return ()
        return [x, y, w, h]

    def rectMerge_sxf(self, rects: []):
        # rects => [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
        rectList = rects.copy()
        rectList.sort()
        new_array = []
        complete = 1
        # 要用while，不能forEach，因爲rectList內容會變
        i = 0
        while i < len(rectList):
            # 選後面的即可，前面的已經判斷過了，不需要重復操作
            j = i + 1
            succees_once = 0
            while j < len(rectList):
                boxa = rectList[i]
                boxb = rectList[j]
                # 判斷是否有重疊，注意只針對水平＋垂直情況，有角度旋轉的不行
                if self.checkOverlap(boxa, boxb):  # intersectionBox(boxa, boxb)
                    complete = 0
                    # 將合並後的矩陣加入候選區
                    new_array.append(self.unionBox(boxa, boxb))
                    succees_once = 1
                    # 從原列表中刪除，因爲這兩個已經合並了，不刪除會導致重復計算
                    rectList.remove(boxa)
                    rectList.remove(boxb)
                    break
                j += 1
            if succees_once:
                # 成功合並了一次，此時i不需要+1，因爲上面進行了remove(boxb)操作
                continue
            i += 1
        # 剩餘項是不重疊的，直接加進來即可
        new_array.extend(rectList)

        # 0: 可能還有未合並的，遞歸調用;
        # 1: 本次沒有合並項，說明全部是分開的，可以結束退出
        if complete == 0:
            complete, new_array = self.rectMerge_sxf(new_array)
        return complete, new_array

    def testRectMerge(self):
        box = [[20, 20, 20, 20], [100, 100, 100, 100], [60, 60, 50, 50], [50, 50, 50, 50]]
        _, res = self.rectMerge_sxf(box)
        print(res)
        print(box)

        img = np.ones([256, 256, 3], np.uint8)
        for x,y,w,h in box:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
        cv2.imshow('origin', img)

        img = np.ones([256, 256, 3], np.uint8)
        for x,y,w,h in res:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
        cv2.imshow('after', img)

        cv2.waitKey(0)
        exit()


    def watershed_algorithm(self, image, preprocess=True, show=True):
        '''分割圖像，返回矩形坐標信息'''
        #image = cv2.imread('/home/sxf/Desktop/pictures/new/a15.jpg', cv2.IMREAD_COLOR)
        origin = image.copy()
        # 边缘保留滤波EPF  去噪. (均值迁移滤波pyrMeanShiftFiltering)
        # 中和色彩分布相近的颜色，平滑色彩细节，侵蚀掉面积较小的颜色区域
        # sp，定义的漂移物理空间半径大小, 物理空间上坐标2个—x、y, 即窗口大小；
        # sr，定义的漂移色彩空间半径大小, 色彩空间上坐标3个—R、G、B, 即像素差值范围；
        # maxLevel=1，定义金字塔的最大层数；
        # termcrit，定义迭代次数满足终止
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        equalizeHist_before = image.copy()
        if preprocess:
            image = equalizeHistOfColor(image)  # 0.0022
        equalizeHist = image.copy()  # 0.0002
        #image = claheOfColor(image)
        clahe = image.copy()
        #t1 = time.time()
        if preprocess:
            image = cv2.pyrMeanShiftFiltering(image, sp=10, sr=10)  # 要准：sp=10, sr=10; 要快：sp=2, sr=4
        #print('>> (pyrMeanShiftFiltering)', (time.time()-t1)*1000)
        pyrMeanShiftFiltering = image.copy()

        # 转成灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (3, 3), 1.6)

        # 得到二值图像   自适应阈值
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # _INV # 0.0014
        #print('>> (OTSU)', (time.time() - t1) * 1000)
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3.0)
        #ret, binary = cv2.threshold(255-self.sobel_sxf(gray, 3), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # _INV # 0.0014
        binary_image = binary.copy()

        # 直方图
        if show:
            '''
            # 灰度
            fig, axes = pylab.subplots(1, 2, figsize=(20, 10))
            axes[0].imshow(gray, interpolation='nearest')
            axes[0].axis('off'),
            axes[1].hist(gray.ravel(), bins=np.arange(0, 256), range=[0, 256])
            axes[1].set_xlim(0, 256)
            axes[1].set_title('histogram of gray values')
            '''
            # 亮度
            fig, axes = pylab.subplots(2, 3, figsize=(20, 30))
            img_hsv = cv2.cvtColor(equalizeHist, cv2.COLOR_BGR2HSV)[:, :, 2]
            img_hsv_before = cv2.cvtColor(equalizeHist_before, cv2.COLOR_BGR2HSV)[:, :, 2]
            img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2RGB)
            img_hsv_before = cv2.cvtColor(img_hsv_before, cv2.COLOR_BGR2RGB)

            axes[0][0].imshow(cv2.cvtColor(equalizeHist_before, cv2.COLOR_BGR2RGB), interpolation='nearest')
            axes[0][0].set_title('均衡前-图像')
            axes[0][0].axis('off')
            axes[0][1].imshow(img_hsv_before, interpolation='nearest')
            axes[0][1].set_title('均衡前-V通道')
            axes[0][1].axis('off')
            axes[0][2].hist(img_hsv_before.ravel(), bins=np.arange(0, 256), range=[0, 256])
            axes[0][2].set_xlim(0, 256)
            axes[0][2].set_title('均衡前-V通道直方图')

            axes[1][0].imshow(cv2.cvtColor(equalizeHist, cv2.COLOR_BGR2RGB), interpolation='nearest')
            axes[1][0].set_title('均衡后-图像')
            axes[1][0].axis('off'),
            axes[1][1].imshow(img_hsv, interpolation='nearest')
            axes[1][1].set_title('均衡后-V通道')
            axes[1][1].axis('off')
            axes[1][2].hist(img_hsv.ravel(), bins=np.arange(0, 256), range=[0, 256])
            axes[1][2].set_xlim(0, 256)
            axes[1][2].set_title('均衡后-V通道直方图')

            # HSV
            color = ('b', 'g', 'r')
            fig, axes = pylab.subplots(2, 2, figsize=(20, 20))
            axes[0][0].imshow(cv2.cvtColor(equalizeHist_before, cv2.COLOR_BGR2RGB), interpolation='nearest')
            axes[0][0].axis('off'),
            axes[0][0].set_title('均衡前-图像')
            axes[0][1].set_title('均衡前－RGB三通道分量分布')
            axes[0][1].set_xlim([0, 256])
            for i, col in enumerate(color):
                histr = cv2.calcHist([cv2.cvtColor(equalizeHist_before, cv2.COLOR_BGR2RGB)], [i], None, [256], [0, 256])
                axes[0][1].plot(histr, color=col)
            axes[1][0].imshow(cv2.cvtColor(equalizeHist, cv2.COLOR_BGR2RGB), interpolation='nearest')
            axes[1][0].axis('off'),
            axes[1][0].set_title('均衡后-图像')
            axes[1][1].set_title('均衡后－RGB三通道分量分布')
            axes[1][1].set_xlim([0, 256])
            for i, col in enumerate(color):
                histr = cv2.calcHist([cv2.cvtColor(equalizeHist, cv2.COLOR_BGR2RGB)], [i], None, [256], [0, 256])
                axes[1][1].plot(histr, color=col)

        # 二值化
        if show:
            ret_raw, binary_raw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # _INV # 0.0014
            fig, axes = pylab.subplots(1, 3, figsize=(20, 30))
            axes[0].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
            axes[0].set_title('原图像灰度图')
            axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(binary_raw, cv2.COLOR_BGR2RGB))
            axes[1].set_title('普通二值化')
            axes[1].axis('off')
            axes[2].imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
            axes[2].set_title('OTSU二值化')
            axes[2].axis('off')

        '''
        # 计算每个二进制像素到最接近的零像素的精确欧几里德距离，然后在这个距离图中找到峰值
        D = ndimage.distance_transform_edt(binary)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=binary)
        # 对局部峰值进行连接组件分析，使用8连通性，然后执行分水岭算法
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=binary)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        '''

        # 形态学操作   获取结构元素  开操作,消除噪声
        # 开运算可参考：https://blog.csdn.net/yukinoai/article/details/86762342
        #t1 = time.time()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # (3, 3)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel=kernel, iterations=2)  # 0.0015, 調整這裏的iterations可以控制前景精度
        # 确定背景区域, 膨胀. 背景是一个不属于任何目标的部分。
        # 膨胀可以将对象的边界延伸到背景中去。这样由于边界区域被去处理，我们就可以知道那些区域肯定是前景，那些肯定是背景。
        # 膨胀操作可参考：https://blog.csdn.net/yukinoai/article/details/86762342
        sure_bg = cv2.dilate(opening, kernel, iterations=2)  # 0.0007

        # 距离变换, 确定前景区域. 前景标注是每一个目标中的一个连接的区域。
        #dist_out = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)  # 5
        #dist_out = cv2.normalize(dist_out, 0, 1.0, cv2.NORM_MINMAX)
        #cv2.imshow('distance-', dist_out)
        #ret, sure_fg = cv2.threshold(dist_out, dist_out.max() * 0.2, 255, cv2.THRESH_BINARY)
        sure_fg = cv2.erode(opening, kernel, iterations=2)  # 0.0007, sure foreground area
        #print('>> (morphologyEx)', (time.time() - t1) * 1000)
        # 寻找未知区域
        #t1 = time.time()
        surface_fg = np.uint8(sure_fg)    # 转成8位整型
        unkonown = cv2.subtract(sure_bg, surface_fg)  # 找到位置区域
        # Marker labelling
        ret, markers = cv2.connectedComponents(surface_fg)  # 连通区域
        #print('>> (connectedComponents)', (time.time() - t1) * 1000)

        # 分水岭变换
        # 所有标签加一，以确保背景不是0而是1
        markers = markers + 1
        # 用0标记未知区域
        markers[unkonown == 255] = 0
        # 实施分水岭算法了。标签图像将会被修改，边界区域的标记将变为 -1
        # skimage.segmentation.watershed
        markers = cv2.watershed(image, markers=markers)
        # 被标记的区域
        image[markers == -1] = [0, 0, 255]
        result = image.copy()

        if show:
            fig, axes = pylab.subplots(figsize=(10, 6))
            a = axes.imshow(markers, cmap=plt.cm.hot, interpolation='nearest')
            plt.colorbar(a)
            axes.set_title('markers hot'), axes.axis('off')

        image_binary = np.zeros(binary.shape, np.uint8)
        image_binary[markers == -1] = 255

        # 遍历marks每一个元素值，对每一个区域进行填充
        #t1 = time.time()
        image_binary_fill = np.zeros(image_binary.shape, dtype=np.uint8)
        image_binary_fill[markers > 1] = 255
        #print('>> (image_binary_fill)', (time.time() - t1) * 1000)

        # image_binary = cv2.GaussianBlur(image_binary, (5, 5), 3)
        # cv2.imshow('image_binary_Gauss', image_binary)
        # connectivity: 2/4连通还是8连通
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_binary_fill, connectivity=8, ltype=None)  # 0.0029
        # 连通域数量
        #print('num_labels = ', num_labels)
        # 连通域的信息：对应各个轮廓的x、y、width、height和面积
        #print('stats = ', stats)
        # 连通域的中心点
        #print('centroids = ', centroids)
        # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
        #print('labels = ', labels)
        imageWithRect = np.array(image)
        # 過濾面積小於limit_area個像素的连通域
        #limit_area = 10  # 10
        filter_stats = []
        line_width = 4
        for i in range(num_labels):
            x, y, w, h, area = stats[i]
            #if area >= limit_area:
            filter_stats.append((x, y, w, h))
            if show:
                imageWithRect = cv2.rectangle(imageWithRect, (x, y), (x + w, y + h), (0, 255, 0), line_width)

        ih, iw, _ = image.shape
        # 圖像邊界值
        border = 0
        rectList = [[x, y, w, h] if (w + x + border <= iw and h + y + border <= ih) else [] for x, y, w, h in filter_stats]
        rectList = list(filter(lambda x: x, rectList))
        rectList = sorted(rectList)[1:]  # 前1個是整個圖的，需要去除，不然就只有一個大框
        _, rectListNew = self.rectMerge_sxf(rectList)  # 0.0004

        margin = 2  # 矩形框四周的擴大量
        # 過濾面積小於limit_area個像素的连通域
        limit_area = 10
        filter_rectNew = []
        # 周圍擴大一點，可能合並的框沒有完全包含目標
        for x, y, w, h in rectListNew:
            area = w * h
            if area >= limit_area:
                x = x - margin if x - margin > 0 else 0
                y = y - margin if y - margin > 0 else 0
                w = w + margin * 2 if x + w + margin * 2 < iw else iw-x
                h = h + margin * 2 if y + h + margin * 2 < ih else ih-y
                filter_rectNew.append([x, y, w, h])

        # _, filter_rectNew = self.rectMerge_sxf(filter_rectNew)  # 0.0004

        if show:
            imageWithRectNew = np.array(image)
            for x, y, w, h in filter_rectNew:
                imageWithRectNew = cv2.rectangle(imageWithRectNew, (x, y), (x+w, y+h), (0, 255, 255), line_width)
                #cv2.putText(imageWithRectNew, str((x, y, x+w, y+h)), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        '''
        # 计算连通域数目
        # 寻找图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次计算连通域数目
        contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 32位有符号整数类型
        marks = np.zeros(image_binary.shape[:2], np.int32)
        # 绘制每一个轮廓
        for index in range(len(contours)):
            marks = cv2.drawContours(marks, contours, index, (index, index, index), -1, 8, hierarchy)
        # 查看, 使用线性变换转换输入数组元素成8位无符号整型。
        markerShows = cv2.convertScaleAbs(marks)
        cv2.imshow('markerShows', markerShows)
        '''

        if show:
            segmentation = ndi.binary_fill_holes(markers - 1)
            labeled_image, _ = ndi.label(segmentation)
            image_label_overlay = label2rgb(labeled_image, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), bg_label=0)
            fig, axes = pylab.subplots(1, 2, figsize=(20, 6), sharey=True)
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=pylab.cm.gray, interpolation='nearest')
            axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
            axes[0].set_title('binary_fill_holes')
            axes[1].imshow(image_label_overlay, interpolation='nearest')
            axes[1].set_title('label2rgb')
            for a in axes:
                a.axis('off')
            pylab.tight_layout()

        if show:
            pylab.show()
            cv2.imshow('origin'         , origin)
            cv2.imshow('claheOfColor'   , clahe)
            cv2.imshow('equalizeHist'   , equalizeHist)
            cv2.imshow('pyrMeanShiftFiltering', pyrMeanShiftFiltering)
            cv2.imshow('binary image'   , binary_image)
            cv2.imshow('opening'        , opening)
            cv2.imshow('sure background', sure_bg)
            cv2.imshow('sure front'     , sure_fg)
            cv2.imshow('unkonown(bg-fg)', unkonown)
            cv2.imshow('result'         , result)
            cv2.imshow('image_binary'   , image_binary)
            cv2.imshow('image_binary_fill', image_binary_fill)
            cv2.imshow('imageWithRect'  , imageWithRect)
            cv2.imshow('imageWithRectNew', imageWithRectNew)
            self.drawColor(markers, image)
            cv2.waitKey(0)

        return filter_rectNew

    def roiImage(self, image, rectList):
        '''根據矩形坐標信息，截取圖像指定區域'''
        imgNew = np.zeros(image.shape, np.uint8)
        for x,y,w,h in rectList:
            if len(image.shape) == 3:
                imgNew[y:y+h, x:x+w, :] = image[y:y+h, x:x+w, :]
            else:
                imgNew[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        return imgNew

    def process(self, image, gray=False, preprocess=True, show=False):
        rectList = self.watershed_algorithm(image, preprocess=preprocess, show=show)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgNew = self.roiImage(image, rectList)  #   [(0,0,850,680),]
        return imgNew

    def binary_thresholded(self, img):
        # Transform image to gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        # Scale result to 0-255
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sx_binary = np.zeros_like(scaled_sobel)
        # Keep only derivative values that are in the margin of interest
        sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

        # Detect pixels that are white in the grayscale image
        white_binary = np.zeros_like(gray_img)
        white_binary[(gray_img > 200) & (gray_img <= 255)] = 1

        # Convert image to HLS
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:, :, 0]
        S = hls[:, :, 2]
        sat_binary = np.zeros_like(S)
        # Detect pixels that have a high saturation value
        sat_binary[(S > 90) & (S <= 255)] = 1

        hue_binary = np.zeros_like(H)
        # Detect pixels that are yellow using the hue component
        hue_binary[(H > 10) & (H <= 25)] = 1

        # Combine all pixels detected above
        binary_1 = cv2.bitwise_or(sx_binary, white_binary)
        binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
        binary = cv2.bitwise_or(binary_1, binary_2)
        #plt.imshow(binary, cmap='gray')

        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Draw figure for binary images
        f, axarr = plt.subplots(1,6)
        f.set_size_inches(25, 8)
        axarr[0].imshow(img)
        axarr[1].imshow(sx_binary, cmap='gray')
        axarr[2].imshow(white_binary, cmap='gray')
        axarr[3].imshow(sat_binary, cmap='gray')
        axarr[4].imshow(hue_binary, cmap='gray')
        axarr[5].imshow(binary, cmap='gray')
        axarr[0].set_title("Undistorted Image")
        axarr[1].set_title("x Sobel Derivative")
        axarr[2].set_title("White Threshold")
        axarr[3].set_title("Saturation Threshold")
        axarr[4].set_title("Hue Threshold")
        axarr[5].set_title("Combined")
        axarr[0].axis('off')
        axarr[1].axis('off')
        axarr[2].axis('off')
        axarr[3].axis('off')
        axarr[4].axis('off')
        axarr[5].axis('off')
        '''

        return binary


if __name__ == '__main__':
    from calibration.perspective import Perspective

    image = cv2.imread('/home/sxf/Desktop/my/pictures/ablation/sa/a2.jpg', cv2.IMREAD_COLOR)
    perspective = Perspective()
    segment = Segment()
    rectList = segment.watershed_algorithm(image, show=False)
    imgNew = segment.roiImage(image, rectList)
    cv2.imshow('imgNew', imgNew)
    # cv2.imwrite('output.jpg', imgNew)

    img_train_raw = perspective.ipmImage(imgNew, 0, 0, 0, np.array([[0, 0, 1]]).T)
    cv2.imshow('image', image)
    cv2.imshow('perspective', img_train_raw)

    #img_train_resize = cv2.resize(img_train_raw, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('img_train_resize', img_train_resize)

    cv2.waitKey(0)
