import os
from utils.library import buildGaussianPyramid, ComputePatches
from model import NetFeature2D
from glob import glob
import cv2
import numpy as np

def draw_matches(img1, img2, matched_points1, matched_points2):
    def ensure_color(img):
        """确保图像是三通道的彩色图像。如果是灰度图，转换为彩色图像。"""
        if len(img.shape) == 2:  # 灰度图（只有高度和宽度）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为BGR彩色图像
        return img

    img1 = ensure_color(img1)
    img2 = ensure_color(img2)
    # 创建一个新图像，宽度为两幅图像宽度之和，高度为两者之间的最大值
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')

    # 将两幅图像放置在新图像上
    new_image[:h1, :w1] = img1
    new_image[:h2, w1:w1 + w2] = img2

    # 对每对匹配的关键点，在新图像上画线
    for p1, p2 in zip(matched_points1, matched_points2):
        start_point = (int(p1[0]), int(p1[1]))
        end_point = (int(p2[0] + w1), int(p2[1]))
        cv2.line(new_image, start_point, end_point, (0, 0, 255), 1)
        cv2.circle(new_image, start_point, 2, (0, 255, 0), -1)
        cv2.circle(new_image, end_point, 2, (255, 0, 0), -1)

    # 在左上角显示带有白色背景的黑色文字匹配数量
    matches_count = len(matched_points1)
    text = f"Matches: {matches_count}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_offset_x, text_offset_y = 10, 30
    box_coords = ((text_offset_x, text_offset_y + 10), (text_offset_x + text_width, text_offset_y - text_height - 10))
    cv2.rectangle(new_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(new_image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return new_image

def simple_eval(image0_path, image1_path):
    hynet = NetFeature2D(do_cuda=True, DLColor=True, mode=1)
    image0_name = image0_path.split("/")[-1].split(".")[0]
    image1_name = image1_path.split("/")[-1].split(".")[0]
    print(f'>> {image0_name}/{image1_name} <<')
    image0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    # 这里可以改为RootSIFT
    sift = cv2.SIFT_create(
        nfeatures=None,
        nOctaveLayers=3,
        contrastThreshold=0.006,
        # edgeThreshold=-10,
        sigma=1.6
    )
    kpts0, decs0 = sift.detectAndCompute(image0.astype('uint8'), None)
    kpts1, decs1 = sift.detectAndCompute(image1.astype('uint8'), None)
    if len(kpts0)==0 or len(kpts1)==0:
        print(f"图1或图2未检测到关键点")
        return
    pyramid = buildGaussianPyramid(image0.astype('uint8'), 6, graydesc=False)
    pts = ComputePatches(kpts0, pyramid, radius_size=64)
    pts = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in pts]) / 255.0
    kpts0, decs0 = hynet.compute_sift(pts, kpts0, True)
    pyramid = buildGaussianPyramid(image1.astype('uint8'), 6, graydesc=False)
    pts = ComputePatches(kpts1, pyramid, radius_size=64)
    pts = np.array([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA) for p in pts]) / 255.0
    kpts1, decs1 = hynet.compute_sift(pts, kpts1, True)
    # 将关键点转换为NumPy数组（仅取坐标）
    kpts0 = np.array([kp.pt for kp in kpts0])
    kpts1 = np.array([kp.pt for kp in kpts1])
    # 对描述符进行匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(decs0, decs1, k=2)
    if len(matches) == 0:
        print(f"knnMatch结果为0")
        return
    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
    matches = good_matches
    # 根据good_matches的索引，从kpts0和kpts1中找出匹配的关键点坐标
    mkpts0 = np.array([kpts0[m.queryIdx] for m in matches])
    mkpts1 = np.array([kpts1[m.trainIdx] for m in matches])
    try:
        points1 = np.float32(mkpts0)
        points2 = np.float32(mkpts1)
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        print(f"{image1_name} => {len(points1[mask.ravel() == 1])}/{len(matches)}")
        if len(points1[mask.ravel() == 1]) > 0:
            matched_points1 = points1[mask.ravel() == 1]
            matched_points2 = points2[mask.ravel() == 1]
            result_image = draw_matches(image0, image1, matched_points1, matched_points2)
            # cv2.imshow('Matched Points', result_image)
            # cv2.waitKey(1000)
            cv2.imwrite(f"match_{image0_name}_{image1_name}.jpg", result_image)
    except Exception as e:
        print("匹配的点数太少: ", e)

if __name__ == '__main__':
    simple_eval("1.jpg", "2.jpg")
