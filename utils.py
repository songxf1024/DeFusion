import math
import os
import pickle
import time
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
from image_utils import gray
import requests
from numpy import sin, cos
from matcher import BfFeatureMatcher, FeatureMatcherTypes, FlannFeatureMatcher


def hamming_distance(a, b):
    return np.count_nonzero(a != b)

def hamming_distances(a, b):
    return np.count_nonzero(a != b, axis=1)

def l2_distance(a, b):
    return np.linalg.norm(a.ravel() - b.ravel())

def l2_distances(a, b):
    return np.linalg.norm(a - b, axis=-1, keepdims=True)

def descriptor_sigma_mad(des1, des2, descriptor_distances=l2_distances):
    dists = descriptor_distances(des1,des2)
    if len(dists) == 0:
        return np.nan, dists
    dists_median = np.median(dists)     # MAD, approximating dists_median=0
    sigma_mad = 1.4826 * dists_median
    return sigma_mad, dists

def descriptor_sigma_mad_v2(des1, des2, descriptor_distances=l2_distances):
    dists = descriptor_distances(des1,des2)
    dists_median = np.median(dists)
    ads = np.fabs(dists - dists_median) # absolute deviations from median
    sigma_mad = 1.4826 * np.median(ads)
    return sigma_mad, dists_median, dists

def combine_images_horizontally(img1, img2):
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[:h2, w1:w1+w2,:3] = img2
    return img3

def combine_images_vertically(img1, img2):
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((h1+h2, max(w1, w2),3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[h1:h1+h2,:w2,:3] = img2
    return img3

def draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_horizontally(img1,img2)
    h1,w1 = img1.shape[:2]
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i]
        size2 = kps2_sizes[i]
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)    # optic flow style
        cv2.line(img3, (a,b),(c+w1,d), color, 2)  # join corrisponding points
        cv2.circle(img3,(a,b),2, color,-1)
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=int(size1), thickness=1)  # draw keypoint size as a circle
        cv2.circle(img3,(c+w1,d),2, color,-1)
        cv2.circle(img3,(c+w1,d), color=(0, 255, 0), radius=int(size2), thickness=1)  # draw keypoint size as a circle
    return img3

def draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_vertically(img1,img2)
    h1,w1 = img1.shape[:2]
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i]
        size2 = kps2_sizes[i]
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)      # optic flow style
        cv2.line(img3, (a,b),(c,d+h1), color, 1)   # join corrisponding points
        cv2.circle(img3,(a,b),2, color,-1)
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=int(size1), thickness=1)  # draw keypoint size as a circle
        cv2.circle(img3,(c,d+h1),2, color,-1)
        cv2.circle(img3,(c,d+h1), color=(0, 255, 0), radius=int(size2), thickness=1)  # draw keypoint size as a circle
    return img3



def draw_feature_matches(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None, horizontal=True):
    if horizontal:
        return draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)
    else:
        return draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)

def add_ones_1D(x):
    return np.array([x[0], x[1], 1])

def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def compute_hom_reprojection_error(H, kps1, kps2, mask=None):
    if mask is not None:
        mask_idxs = (mask.ravel() == 1)
        kps1 = kps1[mask_idxs]
        kps2 = kps2[mask_idxs]
    kps1_reproj = H @ add_ones(kps1).T
    kps1_reproj = kps1_reproj[:2]/kps1_reproj[2]
    error_vecs = kps1_reproj.T - kps2
    return np.mean(np.sum(error_vecs*error_vecs,axis=1))


class MPlotFigure:
    def __init__(self, img=None, title=None, size=(), scale=1, dpi=100):
        self.dpi = dpi
        self.row = 1
        self.col = 1
        if img:
            self.width = round(img.shape[0] * scale / dpi)
            self.height = round(img.shape[1] * scale / dpi)
            # self.fig = plt.figure(dpi=self.dpi, figsize=(self.height,self.width), tight_layout=True, frameon=False)
        else:
            self.fig = plt.figure(dpi=self.dpi, frameon=False)
        if size:
            self.row = size[0]
            self.col = size[1]
            self.fig.add_gridspec(self.row, self.col)
        if title is not None:
            self.fig.suptitle(title)
        # plt.imshow(img)
        plt.axis('off')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    def add(self, index, img, title):
        ax = self.fig.add_subplot(self.row, self.col, index)
        if title is not None:
            ax.set_title(title, y=-0.15)
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(img)

    # actually show the figures
    @staticmethod
    def show():
        plt.show()

    # make it full screen
    def full_screen(self):
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()


class MatchAndDraw:
    def __init__(self, ratio=0.75):
        self.pts_dst = None
        self.ratio = ratio

    def fittingInliers(self, img1, img2, kps1_matched, kps2_matched, img1_box=None, model_fitting_type='homography', show=True):
        hom_reproj_threshold = 5.0  # 单应性重投影误差阈值:像素内允许的最大重投影误差(将点对视为inlier)
        fmat_err_thld = 3.0  # 基本矩阵估计的阈值:以像素为单位, 从点到极线的最大允许距离(将点对视为inlier)
        mask = None
        H = F = None
        h1, w1 = img1.shape[:2]
        pts_dst = np.array([])
        if kps1_matched.shape[0] > 4:
            if show: print('model fitting for', model_fitting_type)
            if model_fitting_type == 'homography':
                # 如果找到足够多的匹配，则传递它们以找到透视图转换。
                # 一旦我们得到了3x3变换矩阵，我们使用它将queryImage的角变换为trainImage中的相应点。 然后我们在img2上绘制它。
                # N.B.: 只有当视图更改对应于两组关键点之间的适当单应性转换时，才可以适当地应用此方法
                #       e.g.: 关键点位于一个平面上，视图的改变对应于一个纯粹的相机旋转
                # 考慮USAC_DEFAULT(DEGENSAC) 和 USAC_MAGSAC
                if False:  # kps1_matched.shape[0] >= 8:
                    H, mask = cv2.findHomography(kps1_matched, kps2_matched, cv2.USAC_FM_8PTS,
                                                # RANSAC USAC_MAGSAC USAC_FM_8PTS(√) RHO USAC_PARALLEL
                                                ransacReprojThreshold=hom_reproj_threshold)
                else:
                    H, mask = cv2.findHomography(kps1_matched, kps2_matched, cv2.USAC_DEFAULT,
                                                # USAC_DEFAULT  USAC_MAGSAC
                                                ransacReprojThreshold=hom_reproj_threshold,
                                                #confidence=0.999999,
                                                #maxIters=100000
                                                )
            else:
                F, mask = cv2.findFundamentalMat(kps1_matched, kps2_matched, cv2.RANSAC, fmat_err_thld, confidence=0.999)
                n_inlier = np.count_nonzero(mask)
                H = F
            if img1_box is None:
                img1_box = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
            else:
                img1_box = img1_box.reshape(-1, 1, 2)
            if H is None: return H, mask, img2, pts_dst
            pts_dst = cv2.perspectiveTransform(img1_box, H)
            img2 = cv2.polylines(img2, [np.int32(pts_dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            reprojection_error = compute_hom_reprojection_error(H, kps1_matched, kps2_matched, mask)
            if show: print('reprojection error: ', reprojection_error)
        else:
            mask = None
            print('Not enough matches are found for', model_fitting_type)
        return H, mask, img2, pts_dst

    def drawingInliers(self, img1, img2, title,
                       kps1, des1, idx1,
                       kps2, des2, idx2,
                       mask, descriptor_distances=l2_distances, draw_horizontal_layout=True, show=True):
        img_matched_inliers = None
        kpts1 = np.array([x.pt for x in kps1], dtype=np.float32)
        kpts2 = np.array([x.pt for x in kps2], dtype=np.float32)
        kps1_size = np.array([x.size for x in kps1], dtype=np.float32)
        kps2_size = np.array([x.size for x in kps2], dtype=np.float32)
        kps1_matched = kpts1[idx1]
        des1_matched = des1[idx1][:]
        kps1_size = kps1_size[idx1]
        kps2_matched = kpts2[idx2]
        des2_matched = des2[idx2][:]
        kps2_size = kps2_size[idx2]
        sigma_mad, dists = descriptor_sigma_mad(des1_matched, des2_matched, descriptor_distances=descriptor_distances)
        if show: print('3 x sigma-MAD of descriptor distances (all): ', 3 * sigma_mad)

        kps1_matched_inliers = []
        kps2_matched_inliers = []
        kps1_size_inliers = []
        kps2_size_inliers = []
        if mask is not None:
            mask_idxs = (mask.ravel() == 1)
            kps1_matched_inliers = kps1_matched[mask_idxs]
            kps1_size_inliers = kps1_size[mask_idxs]
            des1_matched_inliers = des1_matched[mask_idxs][:]
            kps2_matched_inliers = kps2_matched[mask_idxs]
            kps2_size_inliers = kps2_size[mask_idxs]
            des2_matched_inliers = des2_matched[mask_idxs][:]
            print('inliers: {2}% [{0}/{1}]'.format(len(kps1_matched_inliers), len(kps1_matched), (len(kps1_matched_inliers)/(max(len(kps1_matched), 1)+1e-9)*100)))
            sigma_mad_inliers, dists = descriptor_sigma_mad(des1_matched_inliers, des2_matched_inliers, descriptor_distances=descriptor_distances)
            if show: print('3 x sigma-MAD of descriptor distances (inliers): ', 3 * sigma_mad_inliers)
        img_matched_inliers = draw_feature_matches(img1, img2, kps1_matched_inliers, kps2_matched_inliers, kps1_size_inliers, kps2_size_inliers, draw_horizontal_layout)
        if show:
            img_matched = draw_feature_matches(img1, img2, kps1_matched, kps2_matched, kps1_size, kps2_size, draw_horizontal_layout)
            fig = MPlotFigure(size=(2, 1), title=title)
            fig.add(1, img_matched, "All matches")
            if img_matched_inliers is not None: fig.add(2, img_matched_inliers, "Inlier matches")
            fig.show()
        return len(kps1_matched_inliers), img_matched_inliers

    def findMatches(self, des1, des2, num=128, kps1=None, kps2=None, match_type=FeatureMatcherTypes.FLANN, norm_type=cv2.NORM_L2, force=False, show=True, alph=None, new_match=True):
        if match_type == FeatureMatcherTypes.BF:
            matcher = BfFeatureMatcher(norm_type=norm_type, cross_check=False, ratio_test=self.ratio, type=match_type)
        else:
            matcher = FlannFeatureMatcher(norm_type=norm_type, cross_check=False, ratio_test=self.ratio, type=match_type)
        if new_match and (des1.shape[1] == num*2 or force == True):
            print('使用match_concat_optimize')
            idx1, idx2 = matcher.match_concat_optimize(des1, des2, num=num, ratio_test=self.ratio, cross=True, alph=alph)
        else:
            print('使用match')
            idx1, idx2 = matcher.match(des1, des2, ratio_test=self.ratio)
        return idx1, idx2

    def process(self, img1, kps1, des1, img2, kps2, des2, num, norm_type, match_type, dis_type, title, force=False, show=True, alph=None, new_match=True):
        descriptor_distances = l2_distances if dis_type == 'l2' else hamming_distances
        model_fitting_type = 'homography'
        img1_box = None
        draw_horizontal_layout = True
        if torch.is_tensor(des1): des1 = des1.cpu().numpy()
        if torch.is_tensor(des2): des2 = des2.cpu().numpy()
        idx1, idx2 = self.findMatches(des1, des2, num, kps1, kps2, match_type, norm_type, force=force, show=show, alph=alph, new_match=new_match)
        if show:
            print('kps1: ', len(kps1))
            print('kps2: ', len(kps2))
            print('number of matches: ', len(idx1))
        kpts1 = cv2.KeyPoint_convert(kps1).astype(np.float32)
        kpts2 = cv2.KeyPoint_convert(kps2).astype(np.float32)
        kps1_matched = kpts1[idx1]
        kps2_matched = kpts2[idx2]
        H, mask, img2, pts_dst = self.fittingInliers(img1, img2, kps1_matched, kps2_matched, img1_box, model_fitting_type, show=show)
        self.pts_dst = pts_dst.squeeze()
        matched_inliers, img_matched_inliers = self.drawingInliers(img1, img2, title,
                                                                    kps1, des1, idx1,
                                                                    kps2, des2, idx2,
                                                                    mask, descriptor_distances,
                                                                    draw_horizontal_layout, show=show)
        return len(idx1), matched_inliers, img_matched_inliers

    def segIntersect(self, a1, a2, b1, b2):
        T = np.array([[0, -1], [1, 0]])
        da = np.atleast_2d(a2 - a1)
        db = np.atleast_2d(b2 - b1)
        dp = np.atleast_2d(a1 - b1)
        dap = np.dot(da, T)
        denom = np.sum(dap * db, axis=1)
        num = np.sum(dap * dp, axis=1)
        return (np.atleast_2d(num / denom).T * db + b1).squeeze()

    def getPtsCenter_sxf(self, pts_dst=None):
        p1, p2, p3, p4 = pts_dst if pts_dst is not None else self.pts_dst
        a1 = [(p1[0] + p4[0]) / 2, (p1[1] + p4[1]) / 2]
        a2 = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2]
        return [round((a1[0] + a2[0]) / 2, 2), round((a1[1] + a2[1]) / 2, 2)]

    def getPtsCenter(self):
        '''
        p1---p4
        |  c  |
        p2---p3
        '''
        p1, p2, p3, p4 = self.pts_dst
        a1 = np.array([(p1[0] + p4[0]) / 2, (p1[1] + p4[1]) / 2])
        a2 = np.array([(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2])
        b1 = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
        b2 = np.array([(p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2])
        return self.segIntersect(a1, a2, b1, b2)


def draw_matches(image1, kps1, des1, image2, kps2, des2, title=''):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    cnt = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            matchesMask[i] = [1, 0]
            cnt += 1
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img = cv2.drawMatchesKnn(image1, kps1, image2, kps2, matches, None, **draw_params)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} {cnt} Good Matches", y=-0.1)
    plt.axis("off")
    plt.show()

def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = desc
    if not torch.is_tensor(desc): temp = torch.Tensor(desc).to(device)
    temp = temp / temp.pow(2).sum(dim=1, keepdim=True).add(1e-10).pow(0.5)
    if torch.is_tensor(desc): return temp
    return temp.cpu().numpy()

def desc_L2Norm(desc):
    eps = 1e-10
    desc = torch.Tensor(desc)
    norm = torch.sqrt(torch.sum(desc * desc, dim=1) + eps)
    desc = desc / norm.unsqueeze(-1).expand_as(desc)
    return desc

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def input_norm(x):
    std = np.std(x)
    mean = np.mean(x)
    return (x - mean) / (std + 1e-7)

def rootSIFT(descs, eps=1e-7, l2norm=False):
    # 应用Hellinger核进行l1归一化
    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    # 对每一个元素求平方根
    descs = np.sqrt(descs)
    # 是否进行l2归一化，有些不一致. 在RootSIFT论文中并没有指出需要进行l2归一化，但是在presentation, 却有l2归一化.
    # 也有认为，显式地执行L2规范化是不需要的. 通过采用L1规范，然后是平方根，已经有L2标准化的特征向量，不需要进一步的标准
    if l2norm:
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
        descs = desc_l2norm(descs)
    return descs

def unpackSiftOctave(kpt):
    _octave = kpt.octave
    octave = int(_octave)&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1.0/(1<<octave))
    else:
        scale = float(1<<(-octave))
    return (octave, layer, scale)

def extract_patches_array_sift(pyr, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    patches = []
    flt_epsilon = 1.19209e-07
    firstOctave = -1
    nOctaveLayers = 3
    for kp in kps:
        # TODO: 加上尺度空間
        octave, layer, scale = unpackSiftOctave(kp)
        assert octave >= firstOctave and layer <= nOctaveLayers + 2, 'octave = ' + str(octave) + ', layer = ' + str(layer)
        # opencv中的公式: kpt.size = sigma*powf(2. f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
        step = kp.size * scale * 0.5  # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kp.pt) * scale
        angle = 360.0 - kp.angle
        if (np.abs(angle - 360.0) < flt_epsilon):
            angle = 0.0
        if (octave - firstOctave) * (nOctaveLayers + 3) + layer > len(pyr):
            print((octave - firstOctave) * (nOctaveLayers + 3) + layer)
        img = pyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer]
        r = patch_size//2-0.5
        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)

        A = np.float32([[c, -s], [s, c]]) / step
        Rptf = np.matmul(A, ptf)
        x = Rptf[0] - r
        y = Rptf[1] - r
        A = np.hstack([A, [[-x], [-y]]])
        dim = np.int32(2 * r + 1)
        # 如果要缩小图像，通常推荐使用#INTER_AREA插值效果最好，而要放大图像，通常使用INTER_CUBIC
        patch = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        patches.append(patch.astype(np.float32))
    return patches

def get_patches(pyr, kps, patch_size=32, mag_factor=1):
    num_kps = len(kps)
    patches = []
    # 通过向网络提供完整的patch张量来计算描述符
    if num_kps > 0:
        t = time.time()
        patches = extract_patches_array_sift(pyr, kps, patch_size=patch_size, mag_factor=mag_factor)
        patches = np.asarray(patches)
        # patches = (patches / 255. - 0.443728476019) / 0.20197947209  # hardnet???
        patches = patches / 255.
        if True:
            print('patches.shape:', patches.shape)
            print('patch elapsed: ', time.time() - t)
    return patches

def filterMaxNumDesc(kp, des, MaxNum):
    if 0 < MaxNum < len(kp):
        responses = [k.response for k in kp]
        idxs = np.fliplr(np.reshape(np.argsort(responses), (1, -1))).reshape(-1)
        kpF = []
        desF = np.zeros(shape=(MaxNum, des.shape[1]), dtype=des.dtype)
        for n in range(MaxNum):
            kpF.append(kp[idxs[n]])
            desF[n, :] = des[idxs[n], :]
        return kpF, desF
    else:
        return kp, des

def bgr_detect(detector, img, desc=None, useProvided=False, useGray=False, remove=True):
    '''
        对单个B、G、R通道进行预检测
        合并它们并返回关键点和描述符
    '''
    channels = cv2.split(img)
    if desc == None:
        keypoints, descrs = detector.detectAndCompute(channels[0], None)
    else:
        kps1, descrs = detector.detectAndCompute(channels[0], None, desc, useProvided)
    kps2, desc2 = detector.detectAndCompute(channels[1], None, descrs, useProvided)
    keypoints, descrs = detector.detectAndCompute(channels[2], None, desc2, useProvided)
    if useGray:
        keypoints, descrs = detector.detectAndCompute(gray(img), None, descrs, useProvided)
    if remove:
        keypoints, descrs = removeDuplicatesByResponse(keypoints, descrs)
    return keypoints, np.array(descrs)

def hsv_detect(detector, img, desc=None, useProvided=False, useGray=False, remove=True):
    '''
        Preform detection on the H,S channels
        merge them and return keypoints and descriptors
    '''
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img2)
    if desc == None:
        keypoints, descrs = detector.detectAndCompute(channels[0], None)
    else:
        keypoints, descrs = detector.detectAndCompute(channels[0], None, desc, useProvided)
    keypoints, descrs = detector.detectAndCompute(channels[1], None, descrs, useProvided)
    keypoints, descrs = detector.detectAndCompute(channels[2], None, descrs, useProvided)
    if useGray:
        keypoints, descrs = detector.detectAndCompute(gray(img), None, descrs, useProvided)
    if remove:
        keypoints, descrs = removeDuplicatesByResponse(keypoints, descrs)
    return keypoints, np.array(descrs)

def opponent_detect(detector, img, desc=None, useProvided=False, useGray=False, remove=True):
    o1, o2, o3 = convertBGRtoOpponent(img)
    rgb_matrix = o1 + o2 + o3
    o1 = o1 / rgb_matrix
    o2 = o2 / rgb_matrix
    o3 = o3 / rgb_matrix
    if desc == None:
        keypoints1, descrs1 = detector.detectAndCompute(o1, None)
    else:
        keypoints1, descrs1 = detector.detectAndCompute(o1, None, desc, useProvided)
    keypoints = keypoints1
    descrs = descrs1
    keypoints2, descrs2 = detector.detectAndCompute(o2, None)
    keypoints = np.concatenate([keypoints, keypoints2], axis=0)
    descrs = np.concatenate([descrs, descrs2], axis=0)
    keypoints3, descrs3 = detector.detectAndCompute(o3, None)
    keypoints = np.concatenate([keypoints, keypoints3], axis=0)
    descrs = np.concatenate([descrs, descrs3], axis=0)
    if useGray:
        keypoints4, descrs4 = detector.detectAndCompute(gray(img.copy()), None)
        keypoints = np.concatenate([keypoints, keypoints4], axis=0)
        descrs = np.concatenate([descrs, descrs4], axis=0)
    # keypoints, descrs = removeDuplicates(keypoints, descrs)
    if remove:
        keypoints, descrs = removeDuplicatesByResponse(keypoints, descrs)
    return keypoints, np.array(descrs)

def convertBGRtoOpponent(img):
    '''
        将BGR转换为Opponent色彩空间
        Done in Matlab by:

        casted to float32 values
        values are normalized from 0->1

        o1 = (R-G)/math.sqrt(2)
        o2 = (R+G-2*B)/math.sqrt(6)
        o3 = (R+G+B)/math.sqrt(3)

        the values are renormalized back to
        0->255 and then cast back to uint8
    '''
    channels = cv2.split(img)
    # convert to float first
    B = channels[0].astype(np.float32)
    G = channels[1].astype(np.float32)
    R = channels[2].astype(np.float32)
    # Do the conversion
    o1 = (R - G) / math.sqrt(2)
    o2 = (R + G - 2 * B) / math.sqrt(6)
    o3 = (R + G + B) / math.sqrt(3)
    # normalize to 0.0,1.0
    # cv2.normalize(src=o1, dst=o1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(src=o2, dst=o2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(src=o3, dst=o3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # o1 = (0.5 *  (255+G-R))
    # o2 = (0.25 * (510+R+G-2*B))
    # o3 = (1.0/3.0 * (R+G+B))
    # First renormalize values between 0 and 1
    # cv2.normalize(src=o1,dst=o1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(src=o2,dst=o2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(src=o3,dst=o3,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    # Renormalize values for uint8, so 0->255
    cv2.normalize(src=o1, dst=o1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(src=o2, dst=o2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(src=o3, dst=o3, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # convert to uint8 and return
    return o1.astype(np.uint8), o2.astype(np.uint8), o3.astype(np.uint8)

def boostSalientColours(o1, o2, o3):
    '''
        将opponent信道归一化，然后对每个信道应用权重(0.850,0.524,0.065)
        在Matlab中工作，这里还没有
    '''
    # Convert to float type
    o1 = o1.astype(np.float32)
    o2 = o2.astype(np.float32)
    o3 = o3.astype(np.float32)
    # salient colour boosting
    try:
        o1 = o1 / (o1 + o2 + o3)
    except:
        pass
    try:
        o2 = o2 / (o1 + o2 + o3)
    except:
        pass
    try:
        o3 = o3 / (o1 + o2 + o3)
    except:
        pass
    o1 *= 0.850
    o2 *= 0.524
    o3 *= 0.065
    # Convert back to unsigned char type
    o1 = o1.astype(np.uint8)
    o2 = o2.astype(np.uint8)
    o3 = o3.astype(np.uint8)
    return o1, o2, o3

def hellingerKernel(descs):
    # 首先通过l1 -归一化并取平方根来应用Hellinger kernel
    descs /= (descs.sum(axis=1, keepdims=True) + 1e-7)
    descs = np.sqrt(descs)
    # second L2 normalization not required apparently
    # descs /= (np.linalg.norm(descs, axis=1, ord=2) + 1e-7)
    # renormalize values between 0 and 512
    # descs = cv2.normalize(src=descs,dst=None,alpha=0,beta=512,norm_type=cv2.NORM_MINMAX)
    descs *= 512.0  # How SIFT normalized values, didn't think of this one originally
    return descs

def process_bgr(image, desc=None, nfeatures=4096):
    img = cv2.imread(image)
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    if desc == None:
        keypoints, descriptors = bgr_detect(detector, img)
    else:
        keypoints, descriptors = bgr_detect(detector, img, desc, False)
    return keypoints, descriptors

def process_hsv(image, desc=None, nfeatures=4096):
    img = cv2.imread(image)
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    if desc == None:
        keypoints, descriptors = hsv_detect(detector, img)
    else:
        keypoints, descriptors = hsv_detect(detector, img, desc)
    return keypoints, descriptors

def process_opponent(image, kps=None, nfeatures=4096):
    img = cv2.imread(image)
    detector = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    if kps == None:
        keypoints, descriptors = opponent_detect(detector, img)
    else:
        keypoints, descriptors = opponent_detect(detector, img, kps)
    return keypoints, descriptors

def pickle_keypoints(keypoints, descriptors, filename):
    '''
        在Python Pickle文件中存储关键点，描述符，用于后期检测匹配过程
    '''
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt,
                point.size,
                point.angle,
                point.response,
                point.octave,
                point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    pickle.dump(temp_array, open(filename, "wb"))
    return

def unpickle_keypoints(filename):
    '''
        UnPickle the OpenCV Keypoints and Descriptors
    '''
    array = pickle.load(open(filename, "rb"))
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

def writeSift(name, keypoints, desc):
    '''
        Write Lowe SIFT binary format:

        total_features length_of_descriptors(128 in most cases)
        <y> <x> <scale> <orientation in radians> (Floats)
        <128 x descriptors>                      (unsigned char)

        <y> <x> <scale> <orientation in radians>
        <128 x descriptors>

        <y> <x> <scale> <orientation in radians>
        <128 x descriptors>
        .
        .
        .

    '''
    # print "Saving: ",name+".sift"
    sift = open(name + ".sift", 'w')
    sift.write("%d %d \n" % (desc.shape[0], desc.shape[1]))
    for i in range(0, len(keypoints)):
        kpt = "%f %f %f %f\n" % (
            keypoints[i].pt[1], keypoints[i].pt[0], keypoints[i].size, keypoints[i].angle * math.pi / 180.0)
        sift.write(kpt)
        for j in range(0, len(desc[i])):
            sift.write("%d " % desc[i, j])
            if ((j + 1) % 19 == 0): sift.write("\n")
        sift.write("\n")
    sift.close()
    return

def match_images(kps1, kps2, desc1, desc2):
    '''
        Preform FLANN matching
    '''
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    # Matches
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    # Filter the matches
    p1, p2, kp_pairs = filter_matches(kps1, kps2, raw_matches)
    return p1, p2, kp_pairs

def ransac(p1, p2, kp_pairs):
    '''
        Preform RANSAC to remove outliers
    '''
    if len(p1) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        return kp_pairs
    return kp_pairs

def filter_matches(kp1, kp2, matches, ratio=0.75):
    '''
        基于Lowe's比率测试过滤匹配
    '''
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def write_matFile(f, filename1, filename2, kp_pairs):
    '''写入VisualSFM匹配文件'''
    f.write("%s %s %d\n" % (filename1, filename2, len(kp_pairs)))
    for i in range(len(kp_pairs)): f.write(kp_pairs[i][0].pt[0])
    f.write("\n")
    for i in range(len(kp_pairs)): f.write(kp_pairs[i][1].pt[0])
    f.write("\n")

def removeDuplicates(keypoints, descriptors):
    '''
        去掉重复的keypoints/descriptors
    '''
    seen = {}
    new_keypoints = []
    new_descriptors = []
    for index in range(0, len(keypoints)):
        point = keypoints[index].pt
        if not (point[0], point[1]) in seen:
            seen[point] = []
            new_keypoints.append(keypoints[index])
            new_descriptors.append(descriptors[index])
    print(len(keypoints) - len(new_keypoints), "duplicates removed")
    return new_keypoints, np.array(new_descriptors)

def removeDuplicatesByResponse(keypoints, descriptors):
    '''
        去掉重复的keypoints/descriptors
    '''
    seen = {}
    new_keypoints = []
    new_descriptors = []
    new_index = 0
    for index in range(0, len(keypoints)):
        point = keypoints[index].pt
        if not (point[0], point[1]) in seen:
            seen[point] = new_index
            new_keypoints.append(keypoints[index])
            new_descriptors.append(descriptors[index])
            new_index += 1
        else:
            if keypoints[index].response > new_keypoints[seen[point]].response:
                print(keypoints[index].response, new_keypoints[seen[point]].response)
                new_keypoints[new_index] = keypoints[index]

    print(len(keypoints) - len(new_keypoints), "重復點已刪除")
    return new_keypoints, np.array(new_descriptors)

def appendDescriptors(kps, desc, kps2, desc2):
    kps.extend(kps2)
    desc = np.vstack((desc, desc2))
    return kps, desc

def rotateVec(yaw, pitch, roll, reverse=False, order='ZYX', mode='deg'):
    '''航向(yaw)、俯仰(pitch)、横滚(roll)'''
    if mode == 'deg':
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
    if reverse:
        yaw, roll, pitch = -yaw, -roll, -pitch
    Rz = [[cos(yaw),      sin(yaw),         0               ],
          [-sin(yaw),     cos(yaw),         0               ],
          [0,               0,              1               ]]
    Rx = [[1,               0,              0               ],
          [0,               cos(roll),       sin(roll)     ],
          [0,               -sin(roll),      cos(roll)     ]]
    Ry = [[cos(pitch),      0,              -sin(pitch)  ],
          [0,               1,              0               ],
          [sin(pitch),      0,              cos(pitch)   ]]
    Rx = np.array(Rx)
    Ry = np.array(Ry)
    Rz = np.array(Rz)
    if order.upper() == 'XYZ':
        Rvec = np.dot(np.dot(Rx, Ry), Rz)
    else:
        Rvec = np.dot(np.dot(Rz, Ry), Rx)
    return Rvec

def mergeRandT(Rvec, Tvec):
    temp = np.hstack((Rvec, Tvec))
    temp = np.vstack((temp, [[0, 0, 0, 1]]))
    return temp

def checkCloudGPU():
    #return False
    try:
        res = requests.get('https://myip.ipip.net').text
        print(res)
        #res = socket.gethostbyname(socket.gethostname())
        if '202.38.247.228' in res:
            return True
    except:
        return False
    return False


if __name__ == '__main__':
    image1 = cv2.imread('/home/sxf/Desktop/pictures/new/query.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('/home/sxf/Desktop/pictures/new/a10.jpg', cv2.IMREAD_COLOR)
    image2 = cv2.imread('/home/sxf/Desktop/pictures/location2/resized/4.1.jpg', cv2.IMREAD_COLOR)
    sift = cv2.SIFT_create()
    descriptor = cv2.xfeatures2d.BEBLID_create(5.0)
    kps1 = sift.detect(image1, None)
    kps2 = sift.detect(image2, None)
    kps1, des1 = descriptor.compute(image1, kps1)
    kps2, des2 = descriptor.compute(image2, kps2)
    matchAndDraw = MatchAndDraw()
    matchAndDraw.process(image1, kps1, des1, image2, kps2, des2, cv2.NORM_L2, FeatureMatcherTypes.BF, 'hamming', "Raw Match")
    print(matchAndDraw.getPtsCenter())
    print(matchAndDraw.getPtsCenter_sxf())




