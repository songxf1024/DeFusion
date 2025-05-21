import numpy as np
import cv2
import torch
from enum import Enum
from collections import defaultdict
import heapq


class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     
    FLANN = 2


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=0.7, type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    return None 


class FeatureMatcher(object):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=0.7, type = FeatureMatcherTypes.BF):
        self.type = type 
        self.norm_type = norm_type 
        self.cross_check = cross_check
        self.matches = []
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.matcher_name = ''

    def match(self, des1, des2, ratio_test=None):
        matches = self.matcher.knnMatch(des1, des2, k=2)  #knnMatch(queryDescriptors,trainDescriptors)
        self.matches = matches
        return self.goodMatches(matches, des1, des2, ratio_test)

    def compute_distances_no_loops(self, query, train):
        """
        计算X中的每个测试点与X_train中的每个训练点之间的距离，不使用显式循环。
        """
        te = np.sum(query ** 2, axis=1, keepdims=True)
        tr = np.sum(train ** 2, axis=1, keepdims=True)
        M = np.dot(query, train.T)
        sq = -2 * M + te + tr.T
        dists = np.sqrt(sq)
        return dists

    def normalization(self, data):
        _range = data.max() - data.min()
        return (data - data.min()) / _range

    def cosine_similarity(self, List1, List2, list=False):
        if list:
            result = List2.dot(List1) / (np.linalg.norm(List2, axis=1) * np.linalg.norm(List1))
        else:
            result = np.dot(List1, List2) / (np.linalg.norm(List1) * np.linalg.norm(List2))
        return result

    def check_torch(self, point_dist_sift, point_dist_dl, ratio_test, alph=None):
        auto_alpha = False
        matches = []
        epsilon = 1e-7
        gap = 0.2
        t = ratio_test
        topk = torch.topk(torch.tensor(point_dist_sift, device='cuda'), 2, largest=False).indices.cpu().numpy()
        mc_sift_1 = (topk[:, 0], point_dist_sift[np.arange(point_dist_sift.shape[0]), topk[:, 0]])
        mc_sift_2 = (topk[:, 1], point_dist_sift[np.arange(point_dist_sift.shape[0]), topk[:, 1]])

        for i in range(point_dist_sift.shape[0]):
            a, b = mc_sift_1[1][i], mc_sift_2[1][i]
            m, n = point_dist_dl[i][mc_sift_1[0][i]], point_dist_dl[i][mc_sift_2[0][i]]
            T = (n * t - m) / (a - t * b + n * t - m + epsilon)

            m_up_nt = m > n * t
            a_up_bt = a > b * t

            if (a_up_bt and m_up_nt):
                continue
            if (not a_up_bt and not m_up_nt):
                alph = 0.5 if auto_alpha else alph if alph is not None else 0.5
            elif (a_up_bt and not m_up_nt):
                if auto_alpha:
                    alph = 1 if T > 1 else T - gap
                    alph = 0 if alph < 0 else alph
                elif alph is None:
                    alph = 0.45
            elif (not a_up_bt and m_up_nt):
                if auto_alpha:
                    alph = 1 if T > 1 else T + gap
                    alph = 1 if alph > 1 else alph
                elif alph is None:
                    alph = 0.55
            else:
                pass
            new_m1 = cv2.DMatch(_queryIdx=i, _trainIdx=mc_sift_1[0][i], _distance=a * alph + m * (1 - alph), _imgIdx=0)
            new_m2 = cv2.DMatch(_queryIdx=i, _trainIdx=mc_sift_2[0][i], _distance=b * alph + n * (1 - alph), _imgIdx=0)
            matches.append([new_m1, new_m2])
        return matches

    def match_concat_optimize(self, des1, des2, num=128, ratio_test=None, cross=False, alph=None):
        query1_1 = des1[:, :num]
        train1_2 = des2[:, :num]
        query2_1 = des1[:, num:]
        train2_2 = des2[:, num:]
        query1_1 = cv2.normalize(query1_1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        train1_2 = cv2.normalize(train1_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        query2_1 = cv2.normalize(query2_1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        train2_2 = cv2.normalize(train2_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        point_dist_sift = self.compute_distances_no_loops(query1_1, train1_2)
        point_dist_dl = self.compute_distances_no_loops(query2_1, train2_2)
        point_dist_sift = self.normalization(point_dist_sift)
        point_dist_dl = self.normalization(point_dist_dl)
        matches = []
        if cross:
            matches1 = self.check_torch(point_dist_sift, point_dist_dl, ratio_test, alph=alph)
            matches2 = self.check_torch(point_dist_dl, point_dist_sift, ratio_test, alph=alph)
            if len(matches1) != 0 and len(matches2) != 0:
                matches = np.concatenate((matches1, matches2), axis=0)
            elif len(matches1) == 0:
                matches = matches2
        else:
            matches = self.check_torch(point_dist_sift, point_dist_dl, ratio_test)
        if len(matches) == 0: matches = None
        self.matches = matches
        return self.goodMatchesOneToOne(matches, des1, des2, ratio_test)

    def matchWithCrossCheckAndModelFit(self, des1, des2, kps1, kps2, ratio_test=None, cross_check=True, err_thld=1, info=''):
        idx1, idx2 = [], []          
        if ratio_test is None: 
            ratio_test = self.ratio_test
            
        init_matches1 = self.matcher.knnMatch(des1, des2, k=2)
        init_matches2 = self.matcher.knnMatch(des2, des1, k=2)

        good_matches = []

        for i,(m1,n1) in enumerate(init_matches1):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[m1.trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio_test is not None:
                cond2 = m1.distance <= ratio_test * n1.distance
                cond *= cond2
            if cond:
                good_matches.append(m1)
                idx1.append(m1.queryIdx)
                idx2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        _, mask = cv2.findFundamentalMat(good_kps1, good_kps2, cv2.RANSAC, err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return idx1, idx2, good_matches, mask

    def goodMatchesOneToOne(self, matches, des1=None, des2=None, ratio_test=None):
        idx1, idx2 = [], []
        if ratio_test is None: ratio_test = self.ratio_test
        if matches is not None:         
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue
                # 余弦相似性，進一步過濾
                # cos = self.cosine_similarity(des1[m.queryIdx], des2[m.trainIdx], False)
                # if cos < 0.4: continue
                dist = dist_match[m.trainIdx]
                if dist == float_inf:
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    if m.distance < dist: 
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        idx1[index] = m.queryIdx
                        idx2[index] = m.trainIdx
        return idx1, idx2

    def goodMatchesSimple(self, matches, des1, des2, ratio_test=None):
        idx1, idx2 = [], []   
        if ratio_test is None:
            ratio_test = self.ratio_test            
        if matches is not None: 
            for m,n in matches:
                if m.distance < ratio_test * n.distance:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)                                                         
        return idx1, idx2 

    def goodMatches(self, matches, des1, des2, ratio_test=None): 
        return self.goodMatchesOneToOne(matches, des1, des2, ratio_test)


class BfFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=0.7, type = FeatureMatcherTypes.BF):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)     
        self.matcher_name = 'BfFeatureMatcher'   


class FlannFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=0.7, type = FeatureMatcherTypes.FLANN):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        if norm_type == cv2.NORM_HAMMING:
            # FLANN parameters for binary descriptors 
            FLANN_INDEX_LSH = 6
            self.index_params= dict(algorithm = FLANN_INDEX_LSH,   # Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
                        table_number = 6,      # 12
                        key_size = 12,         # 20
                        multi_probe_level = 1) # 2            
        if norm_type == cv2.NORM_L2: 
            # FLANN parameters for float descriptors 
            FLANN_INDEX_KDTREE = 1
            self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50)   # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)  
        self.matcher_name = 'FlannFeatureMatcher'                                                

