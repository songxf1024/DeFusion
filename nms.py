from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import cv2
import torch
import numpy as np

logger = logging.getLogger(__name__)


def process_diou_nms(keypoints, descriptors, radius=None, iou_thresh=0.3):
    if radius == 0:
        return keypoints, descriptors
    scores = np.array([k.response for k in keypoints]).astype(np.float32)
    if radius:
        x1 = lambda center: center.pt[0] - radius / 2
        y1 = lambda center: center.pt[1] - radius / 2
        x2 = lambda center: center.pt[0] + radius / 2
        y2 = lambda center: center.pt[1] + radius / 2
    else:
        x1 = lambda center: center.pt[0] - center.size / 2
        y1 = lambda center: center.pt[1] - center.size / 2
        x2 = lambda center: center.pt[0] + center.size / 2
        y2 = lambda center: center.pt[1] + center.size / 2
    # 提取x,y坐標
    dets = np.array([[x1(k), y1(k), x2(k), y2(k)] for k in keypoints]).astype(np.float32)

    # 篩選後的關鍵點，注意順序可能變了
    res = diou_nms(dets, scores, iou_thresh)  # , beta=1e5)
    indexes = []
    # 匹配尋找篩選後的點在原來數組中的下標
    for item in res:
        i = np.argwhere((dets[:, 0] == item[0]) & (dets[:, 1] == item[1]) & (dets[:, 2] == item[2]) & (dets[:, 3] == item[3]))
        if i.size:
            indexes.append(i[0][0])
    des_new = descriptors[indexes, :]
    if type(keypoints) != np.ndarray:
        kpt_new = np.array(keypoints)[indexes].tolist()
    else:
        kpt_new = keypoints[indexes]
    return np.array(kpt_new), des_new

def diou_nms_cuda(dets, scores, iou_thresh=None, beta=1.0):
    """DIOU non-maximum suppression.
      diou = iou - 方框中心欧几里得距离的平方 / 最小包围方框对角线的平方
      Reference: https://arxiv.org/pdf/1911.08287.pdf
      Args:
        dets: detection with shape (num, 4) and format [x1, y1, x2, y2].
        iou_thresh: IOU threshold,
        参数β,用于控制对距离的惩罚程度.  当 β趋向于无穷大时，DIoU 退化为 IoU，此时的 DIoU-NMS 与标准 NMS 效果相当。
                                    当 β趋向于 0 时，此时几乎所有中心点与得分最大的框的中心点不重合的框都被保留了。
      Returns:
        numpy.array: Retained boxes.
    """
    iou_thresh = iou_thresh or 0.5
    scores = torch.from_numpy(scores)
    dets = torch.from_numpy(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.argsort(scores, descending=True)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    keep = []
    while (order.shape[0]) > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])


        w = torch.maximum(torch.zeros_like(xx2), xx2 - xx1 + 1)
        h = torch.maximum(torch.zeros_like(yy2), yy2 - yy1 + 1)

        t1 = time.time()
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        print(time.time() - t1)

        smallest_enclosing_box_x1 = torch.minimum(x1[i], x1[order[1:]])
        smallest_enclosing_box_x2 = torch.maximum(x2[i], x2[order[1:]])
        smallest_enclosing_box_y1 = torch.minimum(y1[i], y1[order[1:]])
        smallest_enclosing_box_y2 = torch.maximum(y2[i], y2[order[1:]])

        square_of_the_diagonal = (
                (smallest_enclosing_box_x2 - smallest_enclosing_box_x1) ** 2 +
                (smallest_enclosing_box_y2 - smallest_enclosing_box_y1) ** 2)

        square_of_center_distance = ((center_x[i] - center_x[order[1:]]) ** 2 +
                                     (center_y[i] - center_y[order[1:]]) ** 2)

        # Add 1e-10 for numerical stability.
        diou = iou - torch.pow(square_of_center_distance / (square_of_the_diagonal + 1e-10), beta)
        inds = torch.where(diou <= iou_thresh)[0]
        order = order[inds + 1]
    return dets[keep].cpu().numpy()

def diou_nms(dets, scores, iou_thresh=None, beta=1.0):
    """DIOU non-maximum suppression.
  diou = iou - 方框中心欧几里得距离的平方 / 最小包围方框对角线的平方
  Reference: https://arxiv.org/pdf/1911.08287.pdf
  Args:
    dets: detection with shape (num, 4) and format [x1, y1, x2, y2].
    iou_thresh: IOU threshold,
    参数β,用于控制对距离的惩罚程度.  当 β趋向于无穷大时，DIoU 退化为 IoU，此时的 DIoU-NMS 与标准 NMS 效果相当。
                                当 β趋向于 0 时，此时几乎所有中心点与得分最大的框的中心点不重合的框都被保留了。
  Returns:
    numpy.array: Retained boxes.
  """
    iou_thresh = iou_thresh or 0.5
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        smallest_enclosing_box_x1 = np.minimum(x1[i], x1[order[1:]])
        smallest_enclosing_box_x2 = np.maximum(x2[i], x2[order[1:]])
        smallest_enclosing_box_y1 = np.minimum(y1[i], y1[order[1:]])
        smallest_enclosing_box_y2 = np.maximum(y2[i], y2[order[1:]])

        square_of_the_diagonal = (
                (smallest_enclosing_box_x2 - smallest_enclosing_box_x1) ** 2 +
                (smallest_enclosing_box_y2 - smallest_enclosing_box_y1) ** 2)

        square_of_center_distance = ((center_x[i] - center_x[order[1:]]) ** 2 +
                                     (center_y[i] - center_y[order[1:]]) ** 2)

        # Add 1e-10 for numerical stability.
        diou = iou - np.power(square_of_center_distance / (square_of_the_diagonal + 1e-10), beta)
        inds = np.where(diou <= iou_thresh)[0]
        order = order[inds + 1]
    return dets[keep]


def py_greedy_nms(dets, iou_thr):
    """Pure python implementation of traditional greedy NMS.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        iou_thr (float): Drop the boxes that overlap with current
            maximum > thresh.
    Returns:
        numpy.array: Retained boxes.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_thr)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return dets[keep, :]


def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)


def DIOU(box_a, box_b, delta=0.9):
    inter = np.intersect1d(box_a, box_b)  # box_a.intersect(box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
              (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
              (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    x1 = ((box_a[:, :, 2] + box_a[:, :, 0]) / 2).unsqueeze(2).expand_as(inter)
    y1 = ((box_a[:, :, 3] + box_a[:, :, 1]) / 2).unsqueeze(2).expand_as(inter)
    x2 = ((box_b[:, :, 2] + box_b[:, :, 0]) / 2).unsqueeze(1).expand_as(inter)
    y2 = ((box_b[:, :, 3] + box_b[:, :, 1]) / 2).unsqueeze(1).expand_as(inter)

    t1 = box_a[:, :, 1].unsqueeze(2).expand_as(inter)
    b1 = box_a[:, :, 3].unsqueeze(2).expand_as(inter)
    l1 = box_a[:, :, 0].unsqueeze(2).expand_as(inter)
    r1 = box_a[:, :, 2].unsqueeze(2).expand_as(inter)

    t2 = box_b[:, :, 1].unsqueeze(1).expand_as(inter)
    b2 = box_b[:, :, 3].unsqueeze(1).expand_as(inter)
    l2 = box_b[:, :, 0].unsqueeze(1).expand_as(inter)
    r2 = box_b[:, :, 2].unsqueeze(1).expand_as(inter)
    cr = torch.max(r1, r2)
    cl = torch.min(l1, l2)
    ct = torch.min(t1, t2)
    cb = torch.max(b1, b2)
    D = (((x2 - x1) ** 2 + (y2 - y1) ** 2) / ((cr - cl) ** 2 + (cb - ct) ** 2 + 1e-7))
    out = inter / union - D ** delta
    return out.squeeze(0)


def torch_nms(boxes, scores, iou_threshold):
    # _, idx = scores.sort(0, descending=True)  # descending表示降序
    idx = np.argsort(-scores)  # 降序
    boxes_idx = boxes[idx]
    iou = DIOU(boxes_idx, boxes_idx).triu_(diagonal=1)  # 取上三角矩阵，不包含对角线
    B = iou
    while 1:
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= iou_threshold).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    keep = idx[maxA <= iou_threshold]
    return keep


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]], dtype=np.float32)
    print('greedy result:')
    print(py_greedy_nms(boxes, 0.7))
    print('soft nms result:')
    print(py_soft_nms(boxes, method='gaussian'))

    boxes = np.array([[100, 100, 210, 210],
                      [250, 250, 420, 420],
                      [220, 220, 320, 330],
                      [100, 100, 210, 210],
                      [230, 240, 325, 330],
                      [220, 230, 315, 340]], dtype=np.float32)
    scores = np.array([0.72, 0.8, 0.92, 0.72, 0.81, 0.9], dtype=np.float32)
    res = diou_nms(boxes, scores, 0.3)
    print(res)
