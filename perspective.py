# coding: utf-8

from functools import reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import sin, cos


class CameraParameters:
    # 摄像机标定及畸变参数(OpenCV)
    fx = 604.5184  # 605.05515626
    fy = 609.4305  # 609.8805653
    cx = 397.4317  # 396.57273207   # 相機光軸在圖像坐標系中的偏移量, 一般爲圖像的中心點, pix
    cy = 298.7116  # 297.980264     # 相機光軸在圖像坐標系中的偏移量, 一般爲圖像的中心點, pix
    s = 0               # 坐標軸傾斜參數，默認爲0

    k1 = 0.0406  # 6.16884548e-02   # 徑向畸變
    k2 = -0.0774  # -2.23787158e-01  # 徑向畸變
    p1 = 0.0010  # 8.88726848e-04   # 切向畸變
    p2 = 1.8049e-4  # 1.10260785e-05   # 切向畸變
    k3 = 0   # 徑向畸變

    f = 24       # 等效焦距, mm
    dx = f / fx  # 每個像素在圖像平面的物理尺寸, mm/pix
    dy = f / fy  # 每個像素在圖像平面的物理尺寸, mm/pix


    width = 800
    height = 640
    RGB = 0
    fps = 30

    # 标定矩阵
    K = np.array([[fx, s, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # 标定矩阵的逆
    Kinv = np.array([[1 / fx, 0     , -cx / fx],
                     [0     , 1 / fy, -cy / fy],
                     [0     , 0     , 1       ]])

    # 畸变系数
    DistCoef = np.array([k1, k2, p1, p2, k3])

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

def construct_RotationMatrixHomogenous(rotation_angles):
    assert (type(rotation_angles) == list and len(rotation_angles) == 3)
    RH = np.eye(4, 4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

def getRotationMatrixManual(rotation_angles):
    rotation_angles = list(map(lambda x: np.deg2rad(x), rotation_angles))

    phi = rotation_angles[0]  # around x
    gamma = rotation_angles[1]  # around y
    theta = rotation_angles[2]  # around z

    # X rotation
    Rphi = np.eye(4, 4)
    sp = np.sin(phi)
    cp = np.cos(phi)
    Rphi[1, 1] = cp
    Rphi[2, 2] = Rphi[1, 1]
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    # Y rotation
    Rgamma = np.eye(4, 4)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    Rgamma[0, 0] = cg
    Rgamma[2, 2] = Rgamma[0, 0]
    Rgamma[0, 2] = sg
    Rgamma[2, 0] = -sg

    # Z rotation (in-image-plane)
    Rtheta = np.eye(4, 4)
    st = np.sin(theta)
    ct = np.cos(theta)
    Rtheta[0, 0] = ct
    Rtheta[1, 1] = Rtheta[0, 0]
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    R = reduce(lambda x, y: np.matmul(x, y), [Rphi, Rgamma, Rtheta])
    return R

def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    ptsIn2D = ptsIn[0, :]
    ptsOut2D = ptsOut[0, :]
    ptsOut2Dlist = []
    ptsIn2Dlist = []

    for i in range(0, 4):
        ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
        ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

    pin = np.array(ptsIn2Dlist) + [W / 2., H / 2.]
    pout = (np.array(ptsOut2Dlist) + [1., 1.]) * (0.5 * sidelength)
    pin = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout

def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    # M is to be estimated
    M = np.eye(4, 4)

    fVhalf = np.deg2rad(fV / 2.)
    d = np.sqrt(W * W + H * H)
    sideLength = scale * d / np.cos(fVhalf)
    h = d / (2.0 * np.sin(fVhalf))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    # Translation along Z-axis by -h
    T = np.eye(4, 4)
    T[2, 3] = -h

    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])

    # Projection Matrix 
    P = np.eye(4, 4)
    P[0, 0] = 1.0 / np.tan(fVhalf)
    P[1, 1] = P[0, 0]
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    # pythonic matrix multiplication
    F = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

    # 对于ptsIn和ptsOut, shape应该是1,4,3，因为perspectiveTransform()期望这样的数据。  
    # 在c++中，可通过Mat ptsIn(1,4,CV_64FC3)实现;  
    ptsIn = np.array([[
        [-W / 2., H / 2., 0.], [W / 2., H / 2., 0.], [W / 2., -H / 2., 0.], [-W / 2., -H / 2., 0.]
    ]])
    ptsOut = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)

    # check float32 otherwise OpenCV throws an error
    assert (ptsInPt2f.dtype == np.float32)
    assert (ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)

    return M33, sideLength

def warpImage(src, theta, phi, gamma, scale, fovy, corners=None):
    H, W, Nc = src.shape
    M, sl = warpMatrix(W, H, theta, phi, gamma, scale, fovy)  # 计算变形矩阵
    sl = int(sl)
    print('Output image dimension = {}'.format(sl))
    dst = cv2.warpPerspective(src, M, (sl, sl))  # 进行图像扭曲
    return dst

class Perspective:
    def __init__(self):
        self.M = None
        self.base = 1  # 坐標是亞像素的，不好直接在map搜索，需要轉一下

    def ipm(self, intrinsic_mtx, rot_mtx, trans_mtx, img):
        '''
        :param intrinsic_mtx: 相機內參矩陣
        :param rot_mtx: 旋轉矩陣
        :param trans_mtx: 平移矩陣
        :param img: 待變換圖像
        :return: 變換後的圖像
        '''
        # row = height = Point.y
        # col = width  = Point.x
        w = img.shape[1] - 1  # 宽度
        h = img.shape[0] - 1  # 高度
        # 世界坐标中的地平面z=0，转换为摄像机坐标中的平面const_c = dot(normal_c, point), 这是用来根据(u,v)求(x,y,z)_c的
        normal_c = np.dot(rot_mtx, np.array([0, 0, 1]).reshape(3, 1))
        # [0, 0, 0]是origin_w
        origin_c = np.dot(rot_mtx, np.array([0, 0, 0]).reshape(3, 1)) + trans_mtx
        const_c = np.dot(normal_c.T, origin_c)
        # 圖像的四個角點坐標(xyz)
        corners = np.hstack((np.array([0, 0, 1]).reshape(3, 1),
                             np.array([w, 0, 1]).reshape(3, 1),
                             np.array([w, h, 1]).reshape(3, 1),
                             np.array([0, h, 1]).reshape(3, 1)))
        # 相機內參矩陣的逆
        intrin_inv = np.linalg.inv(intrinsic_mtx)
        # norm = (K^-1 dot Xc) = (R*Xw + T) / Zc
        norm = np.dot(intrin_inv, corners)
        z = const_c / np.dot(normal_c.T, norm)
        # point_c = ((K^-1 dot Xc) * Zc) = R*Xw + T
        point_c = norm * z
        R_inv = np.linalg.inv(rot_mtx)
        # point_w = R^-1 dot (((K^-1 dot Xc) * Zc) - T) = Xw
        point_w = np.dot(R_inv, point_c - trans_mtx)

        xmin = np.min(point_w[0])
        xmax = np.max(point_w[0])
        ymin = np.min(point_w[1])
        ymax = np.max(point_w[1])

        # 坐标平移到中间可视区域
        point_w = point_w[:2] - np.array([xmin, ymin]).reshape(2, 1)
        point_w = point_w * np.array([w, h]).reshape(2, 1) / np.array([xmax - xmin, ymax - ymin]).reshape(2, 1)
        # 目标图像中相应四边形顶点的坐标。
        new_pixel = np.float32(point_w.T)
        # 源图像中四边形顶点的坐标。
        temp_corners = np.float32(corners[:2, :].T)
        # 获取变换矩阵
        Map = cv2.getPerspectiveTransform(temp_corners, new_pixel)
        # 执行图像变换
        # flags=cv2.WARP_INVERSE_MAP+cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS
        result = cv2.warpPerspective(img, Map, (w + 1, h + 1), flags=cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS)
        self.M = Map
        return result

    def ipmImage(self, src, yaw=0, pitch=0, roll=0, Tvec=np.array([[0, 0, 0.0001]]).T):
        Rvec = rotateVec(yaw, pitch, roll, reverse=True, order='ZYX', mode='deg')
        K = CameraParameters.K
        im_out = self.ipm(K, Rvec, Tvec, src)
        #cv2.imshow("Destination Image", src)
        #cv2.imshow("Warped Source Image", im_out)
        #cv2.waitKey(0)
        return im_out

    def calAllIpmPointMap(self, img):
        w = img.shape[1] - 1  # 宽度
        h = img.shape[0] - 1  # 高度
        p_map = {}
        for u in range(w):
            for v in range(h):
                x, y, z = self.ipmPoint(u, v, raw=True)
                p_map[(int(x/z*self.base), int(y/z*self.base))] = z
        return p_map

    def getZfromPointMap(self, x, y, p_map, max_gap=2):
        new_x = int(x * self.base)
        new_y = int(y * self.base)
        if p_map.get((new_x, new_y)):
            return p_map[(new_x, new_y)]
        for step in range(1, max_gap+1):
            if p_map.get((new_x - step, new_y)):
                return p_map[(new_x - step, new_y)]
            if p_map.get((new_x, new_y - step)):
                return p_map[(new_x, new_y - step)]
            if p_map.get((new_x - step, new_y - step)):
                return p_map[(new_x - step, new_y - step)]

    def ipmPoint(self, u, v, raw=False):
        coord = np.dot(self.M, np.array([u, v, 1]).reshape(3, -1)).reshape(-1)
        if raw:
            return coord
        coord /= coord[2]
        return coord[:2]

    def repmPoint(self, x, y, z):
        coord = np.dot(np.linalg.inv(self.M), np.array([x*z, y*z, z]).reshape(3, -1)).reshape(-1)
        return coord[:2]


def test1():
    import os
    src = cv2.imread(os.getcwd() + '/../pictures/location2/resized/4.1.jpg')
    target = cv2.imread(os.getcwd() + '/../pictures/location2/resized/4.1.jpg')

    src = src[..., ::-1]  # BGR to RGB
    target = target[..., ::-1]  # BGR to RGB
    H, W, Nc = src.shape
    plt.imshow(src)

    imgwarped = warpImage(target, 0, -30, 0, 1., 83)
    plt.figure()
    plt.imshow(imgwarped)
    print(target.shape)
    print(imgwarped.shape)

    theta = np.rad2deg(np.arctan2(600, 800))
    print(theta)
    plt.show()

def test2():
    import os
    yaw = 0
    pitch = 0
    roll = -30
    Tvec = np.array([[0, 0, 1]]).T
    perspective = Perspective()
    src = cv2.imread('/home/sxf/Desktop/my/pictures/new/a5.jpg')
    #src = src[300:365, 365:515, :]
    #plt.imshow(src)
    target = perspective.ipmImage(src, yaw, pitch, roll, Tvec)
    plt.figure()
    plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    # test1()
    # test2()
    
    image = cv2.imread('/../pictures/new/a2.jpg', cv2.IMREAD_COLOR)
    perspective = Perspective()
    perspective.ipmImage(image)

    p_map = perspective.calAllIpmPointMap(image)
    x, y = perspective.ipmPoint(380, 400)
    z = perspective.getZfromPointMap(x, y, p_map)
    u, v = perspective.repmPoint(x, y, z)
    print(u, v)




