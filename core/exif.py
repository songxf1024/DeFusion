# -*- coding: utf-8 -*-
# @Time    : 2021/12/24 下午3:19
# @Author  : 小锋学长生活大爆炸
# @FileName: dji_exif.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/sxf1061700625
import os.path
import sys
import matplotlib
import pyexif
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from pyproj import Transformer
from common_utils import checkCloudGPU
# matplotlib.use('TkAgg')

np.set_printoptions(suppress=True)

class DJIExif:
    def __init__(self):
        self.exif = None
        self.tags = None

    def setImage(self, imgPath) -> None:
        self.tags = None  # 更換圖片後需要重置緩存
        self.exif = pyexif.ExifEditor(imgPath)

    def getDictTags(self) -> dict:
        self.tags = self.exif.getDictTags()
        return self.tags

    def convertGPSToSexagesimal(self, ori) -> float:
        '''度分秒形式(40°45´42.757")轉爲六十進制(40.7618770°)'''
        pattern = re.compile(r'[0-9]\d?\.?\d*')
        matchObj = re.findall(pattern, ori)
        d, m, s = map(float, matchObj)
        return d + (m + s / 60) / 60

    def getGPS(self) -> dict:
        '''無人機的ＧＰＳ坐標'''
        GPSLatitude = self.convertGPSToSexagesimal(self.tags['GPSLatitude'])
        GPSLongitude = self.convertGPSToSexagesimal(self.tags['GPSLongitude'])
        GPSLatitudeRef = self.tags['GPSLatitudeRef']
        GPSLongitudeRef = self.tags['GPSLongitudeRef']
        GPSAltitude = float(self.tags['GPSAltitude'].split(' ')[0])
        return {'latitude': GPSLatitude, 'longitude': GPSLongitude, 'latitudeRef': GPSLatitudeRef, 'longitudeRef': GPSLongitudeRef, 'altitude': GPSAltitude}

    def getFOV(self) -> float:
        '''相機的視角'''
        return float(self.tags['FOV'].split(' ')[0])

    def getRelativeAltitude(self) -> float:
        '''相對高度'''
        return float(self.tags['RelativeAltitude'])

    def getAbsoluteAltitude(self) -> float:
        '''絕對高度(海拔)'''
        return float(self.tags['AbsoluteAltitude'])

    def getGimbalDegree(self) -> dict:
        '''雲臺的３個姿態角度'''
        GimbalRollDegree = float(self.tags['GimbalRollDegree'])
        GimbalYawDegree = float(self.tags['GimbalYawDegree'])
        GimbalPitchDegree = float(self.tags['GimbalPitchDegree'])
        return {'roll': GimbalRollDegree, 'yaw': GimbalYawDegree, 'pitch': GimbalPitchDegree}

    def getDroneDegree(self) -> dict:
        '''雲臺的３個姿態角度'''
        FlightRollDegree = float(self.tags['FlightRollDegree'])
        FlightYawDegree = float(self.tags['FlightYawDegree'])
        FlightPitchDegree = float(self.tags['FlightPitchDegree'])
        return {'roll': FlightRollDegree, 'yaw': FlightYawDegree, 'pitch': FlightPitchDegree}

    def getDroneStatus(self):
        '''無人機的ＧＰＳ坐標＋三個角度＋相機的視角＋相對高度'''
        gps = self.getGPS()
        uavdeg = self.getDroneDegree()
        gimdegree = self.getGimbalDegree()
        fov = self.getFOV()
        altitude = self.getRelativeAltitude()
        return {'gps': gps, 'uavdeg': uavdeg, 'gimdegree': gimdegree, 'fov': fov, 'altitude': altitude}

def draw3DScatter(store, calMean=False, show=False):
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'limegreen', 'orange', 'goldenrod', 'steelblue']
    marker = ['.', 'o', 'v', '^', 's', 'p', '+', 'x', '*', 'd', 'P']
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 7)
        lats = coors[:, 0]
        lons = coors[:, 1]
        hs = coors[:, 3]
        #x, y, z = llaToPlan(lats, lons, hs)
        x, y, z = lats, lons, hs
        ax.scatter(x, y, z, c=color[id - 1], marker=marker[id - 1], label='p' + str(id), linewidths=3)
        if calMean:
            x_m = np.mean(x)
            y_m = np.mean(y)
            z_m = np.mean(z)
            ax.scatter(x_m, y_m, z_m, c=color[id - 1], marker='8', linewidths=2, label='p' + str(id)+'_m')

    # 绘制图例
    ax.legend(loc=1, bbox_to_anchor=(1.2, 1.0))
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z')  # , fontdict={'size': 8, 'color': 'red'}
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    if show:
        x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.yaxis.set_major_formatter(x_formatter)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        #plt.tight_layout()
        # 展示
        plt.show()

def draw2DScatter(store, calMean=False, text=False, show=False):
    # 绘制散点图
    plt.figure()
    color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'limegreen', 'orange', 'goldenrod', 'steelblue']
    marker = ['.', 'o', 'v', '^', 's', 'p', '+', 'x', '*', 'd', 'P']
    max_x = -1e9
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 7)
        lats = coors[:, 0]
        lons = coors[:, 1]
        hs = coors[:, 3]
        # x, y, z = llaToPlan(lats, lons, hs)
        x, y, z = lats, lons, hs
        max_x = max(max_x, np.max(x))
        plt.scatter(x, y, c=color[id - 1], marker=marker[id - 1], label='p' + str(id), linewidths=2)
        if calMean:
            x_m = np.mean(x)
            y_m = np.mean(y)
            plt.scatter(x_m, y_m, c=color[id - 1], marker='8', linewidths=3, label='p' + str(id)+'_m')
        if text:
            for i in range(len(x)):
                x_t, y_t, z_t = x[i], y[i], hs[i]
                plt.text(x_t, y_t, z_t)
    if show:
        # 绘制图例
        plt.legend(loc=1, bbox_to_anchor=(1.0, 1.0))
        plt.xlim(right=max_x + 0.00002)
        plt.xticks(rotation=15)
        plt.tight_layout()
        # 展示
        plt.show()

def statistics(store):
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 6)
        x = coors[:, 0]
        y = coors[:, 1]
        z = coors[:, 2]
        print(np.mean(x), np.var(x), np.std(x))  # 均值, 方差, 标准差
        print(np.mean(y), np.var(y), np.std(y))  # 均值, 方差, 标准差
        print(np.mean(z), np.var(z), np.std(z))  # 均值, 方差, 标准差

def wgs84toWebMercator(lat, lon):
    x = lon*20037508.342789/180
    y = np.log(np.tan((90+lat)*np.pi/360))/(np.pi/180)
    y = y *20037508.34789/180
    return x, y

def wgs84ToMercator(lat, lon, crs_from="epsg:4326", crs_to="epsg:32649", reverse=True):
    # 参数1：WGS84地理坐标系统 对应 4326
    # 参数2：坐标系WKID 广州市 WGS_1984_UTM_Zone_49N 对应 32649
    # 返回:　緯度+經度 => x(N), y(E)
    # 地理坐标系WKID：https://developers.arcgis.com/javascript/3/jshelp/gcs.htm
    # 投影坐标系WKID：https://developers.arcgis.com/javascript/3/jshelp/pcs.htm
    # pyproj官网: https://www.osgeo.cn/pyproj/examples.html
    # UTM地区编号: http://www.dmap.co.uk/utmworld.htm
    # Mercator輸入輸出爲x(E), y(N); 而NED爲x(N), y(E)．注意轉換
    transformer = Transformer.from_crs(crs_from, crs_to)
    e, n = transformer.transform(lat, lon)
    if reverse:
        x, y = n, e
    else:
        x, y = e, n
    return x, y

def llaToPlan(x, y, z, reverse=True):
    res_x, res_y = [], []
    for i in range(len(x)):
        newx, newy = wgs84ToMercator(x[i], y[i], reverse=reverse)
        res_x.append([newx, ])
        res_y.append([newy, ])
    res_x = np.array(res_x).reshape(-1, 1)
    res_y = np.array(res_y).reshape(-1, 1)
    return res_x, res_y, z

def calMean(store):
    res = []
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 7)
        x = coors[:, 0]
        y = coors[:, 1]
        h = coors[:, 3]
        res.append([np.mean(x), np.mean(y), np.mean(h)])
    return res

def calMeanMore(store):
    res = []
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 7)
        x = coors[:, 0]
        y = coors[:, 1]
        z = coors[:, 2]
        h = coors[:, 3]
        roll = coors[:, 4]
        yaw = coors[:, 5]
        pitch = coors[:, 6]
        res.append([np.mean(x), np.mean(y), np.mean(z), np.mean(h), np.mean(roll), np.mean(yaw), np.mean(pitch)])
    return res

def calSmallestOne(store):
    def cmp(a, b):
        a_roll = a[4]
        #a_yaw = a[5]
        a_pitch = a[6]
        b_roll = b[4]
        #b_yaw = b[5]
        b_pitch = b[6]
        d_a = np.linalg.norm([a_roll, a_pitch])
        d_b = np.linalg.norm([b_roll, b_pitch])
        if d_b < d_a:
            return 1
        if d_a < d_b:
            return -1
        return 0
    res = []
    for id in store.keys():
        coors = np.array(store.get(id)).reshape(-1, 7)
        coors_new = sorted(coors, key=cmp_to_key(cmp))
        res.append(coors_new[0].tolist())
    return res

def calOne(store, index):
    res = []
    for id in store.keys():
        if id == 1:
            index = 0
        coors = np.array(store.get(id)).reshape(-1, 7)
        x = coors[index, 0]
        y = coors[index, 1]
        z = coors[index, 2]
        h = coors[index, 3]
        roll = coors[index, 4]
        yaw = coors[index, 5]
        pitch = coors[index, 6]
        res.append([x, y, z, h, roll, yaw, pitch])
    return res

def save(path, content):
    with open(path, 'w+') as f:
        f.write('# GPS緯度, GPS經度, GPS高程, 氣壓計高度, GPS roll, GPS yaw, GPS pitch' + '\n')
        for item in content:
            f.write(str(item) + '\n')

from functools import cmp_to_key
def calculate_dji_exif(root_path, show=False, only_get_path=False, index=None):
    def cmp(a_path, b_path):
        a = int(a_path.split(os.sep)[-1].split('.')[0])
        b = int(b_path.split(os.sep)[-1].split('.')[0])
        if b < a:
            return 1
        if a < b:
            return -1
        return 0
    current_path = os.path.dirname(__file__)
    store_path = os.path.join(current_path, 'loc.txt')
    if only_get_path:
        return store_path
    exif = DJIExif()
    lists = sorted(glob.glob(os.path.join(root_path, '*.JPG')), key=cmp_to_key(cmp))
    store = {}
    storeAll = {}
    for img in lists:
        print(img)
        id, num = img.split('/')[-1][:-4].split('.')
        id = int(id)
        num = int(num)
        exif.setImage(img)
        exif.getDictTags()
        status = exif.getDroneStatus()
        print(status)
        x = status['gps']['latitude']
        y = status['gps']['longitude']
        z = status['gps']['altitude']
        h = status['altitude']
        roll_u = status['uavdeg']['roll']
        yaw_u = status['uavdeg']['yaw']
        pitch_u = status['uavdeg']['pitch']
        roll_c = status['gimdegree']['roll']
        yaw_c = status['gimdegree']['yaw']
        pitch_c = status['gimdegree']['pitch']
        if store.get(id):
            store.get(id).append([x, y, z, h, roll_u, yaw_u, pitch_u])
        else:
            store[id] = [[x, y, z, h, roll_u, yaw_u, pitch_u]]
    if index is not None:
        res = calOne(store=store, index=index-1)
    else:
        #res = calSmallestOne(store=store)
        res = calMeanMore(store)
    print(res)

    save(store_path, res)
    if show:
        # statistics(store)
        draw3DScatter(store, calMean=False, show=show)
        draw2DScatter(store, calMean=False, text=False, show=show)
        # 展示
        plt.show()
    return store_path

if __name__ == '__main__':
    # img_path = '/home/sxf/Desktop/pictures/location/DJI_0203.JPG'
    # exif = DJIExif()
    # exif.setImage(img_path)
    # print(exif.getDictTags())
    # print(exif.getRelativeAltitude())
    # print(exif.getAbsoluteAltitude())
    # print(exif.getFOV())
    # print(exif.getGimbalDegree())
    # print(exif.getDroneDegree())
    # print(exif.getGPS())
    # print(exif.getDroneStatus())
    calculate_dji_exif('./pictures/test/2.png', show=True)

