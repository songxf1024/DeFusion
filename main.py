import cv2
import matplotlib
import numpy as np
from car_hynet.utils.library import buildGaussianPyramid, ComputePatches
from matcher import FeatureMatcherTypes
from nms import process_diou_nms
from image_utils import preprocess
from segmentation import Segment
from perspective import Perspective
from car_hynet.model import NetFeature2D
from utils import MatchAndDraw, rootSIFT, filterMaxNumDesc, desc_l2norm, draw_matches
matplotlib.use('TkAgg')
# matplotlib.use('Agg')


# ----------------------------------------Common Start----------------------------------------------------- #
use_root = 'root'                              # SIFT特徵點規範化: l2/root/None
use_affine = False                             # 使用仿射檢測
use_affineExt = True                           # 先仿射變換糾正後再檢測，與Affine互斥
# ------------------------------------ #
# affineExt啓用後，注意修改這裏！！！
yaw = 0
pitch = -0
roll = -0
Tvec = np.array([[0, 0, 1]]).T
# ------------------------------------ #
use_nms = False                                # 使用非最大值抑制(√仿射時效果好,nfeatures=300都行;小目標看情況)
iou_radius = 3                                 # IOU半徑
iou_thresh = 0.4                               # IOU判斷閾值
max_desc_num = 5000                            # 過濾後的最大特徵點數
nfeatures = 500 if use_affine else 20000       # 要求SIFT檢測出的特徵點數 5000, None表示不作具體限制
nOctaveLayers = 3                              # 空間金字塔每個Octave的層數
maxoctaves = 4                                 # 空間金字塔的Octave數
contrastThreshold = -10000
edgeThreshold = -10000
nndr_ratio = 0.85                              # 比例過濾值
# ----------------------------------------Common End----------------------------------------------------- #

class Matcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=1.6)
        self.asift = cv2.AffineFeature_create(backend=self.sift)
        self.segment = Segment()
        self.net = NetFeature2D(do_cuda=True, DLColor=True, mode=1)
        self.perspective = Perspective()
        self.matchAndDraw = MatchAndDraw(nndr_ratio)

    def process(self, img_query_path, img_train_path):
        img_query_raw = cv2.imread(img_query_path, cv2.IMREAD_COLOR)
        img_train_raw = cv2.imread(img_train_path, cv2.IMREAD_COLOR)
        img_query_raw = cv2.cvtColor(img_query_raw, cv2.COLOR_BGR2RGB)
        img_train_raw = cv2.cvtColor(img_train_raw, cv2.COLOR_BGR2RGB)

        img_query = preprocess(img_query_raw)
        img_train = preprocess(img_train_raw)
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        img_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2RGB)
        img_query_raw = self.segment.process(img_query_raw, gray=False)
        img_train_raw = self.segment.process(img_train_raw, gray=False)
        img_query = self.segment.process(img_query, gray=True)
        img_train = self.segment.process(img_train, gray=True)

        if use_affine:
            kps_query, des_query = self.asift.detectAndCompute(img_query, None)
            kps_train, des_train = self.asift.detectAndCompute(img_train, None)
        elif use_affineExt:
            kps_query, des_query = self.sift.detectAndCompute(img_query, None)
            img_train = self.perspective.ipmImage(img_train, yaw, pitch, roll, Tvec)
            img_train_raw = self.perspective.ipmImage(img_train_raw, yaw, pitch, roll, Tvec)
            kps_train, des_train = self.sift.detectAndCompute(img_train, None)
            p_map = self.perspective.calAllIpmPointMap(img_train)
        else:
            kps_query, des_query = self.sift.detectAndCompute(img_query, None)
            kps_train, des_train = self.sift.detectAndCompute(img_train, None)
            print('傳統query：', len(kps_query))

        kps_query, des_query = np.array(kps_query), np.array(des_query)
        des_query = rootSIFT(des_query) if use_root == 'root' else (desc_l2norm(des_query) if use_root == 'l2' else des_query)
        des_train = rootSIFT(des_train) if use_root == 'root' else (desc_l2norm(des_train) if use_root == 'l2' else des_train)
        kps_query, des_query = process_diou_nms(kps_query, des_query, iou_radius, iou_thresh) if use_nms else (kps_query, des_query)
        kps_train, des_train = process_diou_nms(kps_train, des_train, iou_radius, iou_thresh) if use_nms else (kps_train, des_train)
        kps_query, des_query = filterMaxNumDesc(kps_query, des_query, max_desc_num)
        kps_train, des_train = filterMaxNumDesc(kps_train, des_train, max_desc_num)

        pyr1 = buildGaussianPyramid(img_query_raw, maxoctaves + 2, graydesc=False)
        pyr2 = buildGaussianPyramid(img_train_raw, maxoctaves + 2, graydesc=False)
        patches1 = ComputePatches(kps_query, pyr1, radius_size=64)
        patches2 = ComputePatches(kps_train, pyr2, radius_size=64)
        patches1 = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in patches1]) / 255.0
        patches2 = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in patches2]) / 255.0

        kps_query_new, des_query_new = self.net.compute_sift(patches1, kps_query, True)
        kps_train_new, des_train_new = self.net.compute_sift(patches2, kps_train, True)
        des_query_cat = np.concatenate((des_query, des_query_new), axis=1)
        des_train_cat = np.concatenate((des_train, des_train_new), axis=1)

        # ----------------------------------------Match Two Raw Images----------------------------------------------------- #
        self.matchAndDraw.process(img_query, kps_query, des_query, img_train, kps_train, des_train,
                                  128, cv2.NORM_L2, FeatureMatcherTypes.BF, 'l2', "SIFT Match")
        # draw_matches(img_query, kps_query, des_query, img_train, kps_train, des_train, "SIFT")

        # ----------------------------------------Match Two DL Images----------------------------------------------------- #
        self.matchAndDraw.process(img_query_raw, kps_query_new, des_query_new, img_train_raw, kps_train_new, des_train_new,
                                  128, cv2.NORM_L2, FeatureMatcherTypes.BF, 'l2', "Deep Match")
        # draw_matches(img_query, kps_query_new, des_query_new, img_train, kps_train_new, des_train_new, "Deep")

        # ----------------------------------------Match Two Cat Images----------------------------------------------------- #
        self.matchAndDraw.process(img_query_raw, kps_query_new, des_query_cat, img_train_raw, kps_train_new, des_train_cat,
                                  128, cv2.NORM_L2, FeatureMatcherTypes.BF, 'l2', "Fusion Match", alph=0.4)
        # draw_matches(img_query, kps_query_new, des_query_cat, img_train, kps_train_new, des_train_cat, "Fusion")

        print(self.matchAndDraw.pts_dst)
        x, y = self.matchAndDraw.getPtsCenter_sxf()
        z = self.perspective.getZfromPointMap(x, y, p_map) if use_affineExt else None
        if use_affineExt and z: x, y = self.perspective.repmPoint(x, y, z)
        print('最終目標在圖像的中心點: ', x, y, z)
        return x, y


if __name__ == '__main__':
    img_query = 'images/query.jpg'
    img_train = 'images/d2.jpg'
    matcher = Matcher()
    matcher.process(img_query, img_train)




