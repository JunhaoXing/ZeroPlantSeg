import os

import supervision as sv
from PIL import Image
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from attention_map.edit_mask import crop_with_mask, resize2square, mask2bbox, crop_att_map_with_bbox
from attention_map.gdino_attention import load_groundingDINO, get_attention_map, cvt_img2tensor

def get_furthest_points(mask: np.ndarray) -> list:
    """get a pair (list) of points whose distance is the largest in the mask
        when the number of contours != 0, return only points of largest contour 

    Args:
        mask (np.ndarray): whose shape is (H, W)

    Returns:
        max_dist_pair (list): [np.array([x1, y1]), np.array([x2, y2])]
    """
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # shape (N, 1, [x, y])
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    
    # init
    max_dist = -1
    max_dist_pair = None
    for i in range(len(contour)):
        for j in range(i+1, len(contour)):
            dist = np.linalg.norm(contour[i][0] - contour[j][0])
            if dist > max_dist:
                max_dist = dist
                max_dist_pair = [contour[i][0], contour[j][0]]
    
    return max_dist_pair

def sum_on_line(img, p1, p2):
    """calc sum of pixel values on line which runs through p1 and p2
        the line is defined as y = ax + b (0<=y<H)

    Args:
        img (np.ndarray): gray scale image, shape (H, W)
        p1 (list): [x1, y1]
        p2 (list): [x2, y2]

    Returns:
        float: _description_
    """
    h, w = img.shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        len_line = h
        return np.sum(img[:, x1]) / len_line
    elif dy == 0:
        len_line = w
        return np.sum(img[y1, :]) / len_line
    else:
        a = dy / dx
        b = y1 - a * x1
        x_list = []
        y_list = []
        # get each crd of the line
        for x in range(w):
            y = a * x + b
            if round(y) < 0 or round(y) >= h:
                continue
            else:
                x_list.append(round(x))
                y_list.append(round(y))
        len_line = len(x_list)
        return np.sum(img[y_list, x_list]) / len_line

def get_line_points(img, p1=None, p2=None, a=None, b=None):
    """get points on line which runs through p1 and p2 in non-zero region of image plane(img)

    Args:
        img (np.ndarray): the shape is (H, W) or (H, W, 3)
        p1 (_type_): _description_
        p2 (_type_): _description_
    
    Returns:
        [line_x, line_y] (List[List[int], List[int]]): _description_
    """
    h, w = img.shape[:2]
    mask_range = [min(np.nonzero(img)[0]), max(np.nonzero(img)[0]), min(np.nonzero(img)[1]), max(np.nonzero(img)[1])]
    nonzero_region = [(y, x) for (y, x) in zip(np.nonzero(img)[0], np.nonzero(img)[1])]
    
    if p1 is not None and p2 is not None:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        a = dy / dx
        b = y1 - a * x1
        if dx == 0:
            return [x1] * mask_range[1], list(mask_range[1])
        elif dy == 0:
            return list(mask_range[3]), [y1] * mask_range[3]
    elif a is not None and b is not None:
        pass
    else:
        raise ValueError("(p1, p2) or (a, b) must be given")
    
    line_x = []
    line_y = []
    for x in range(mask_range[2], mask_range[3]):
        y = round(a * x + b)
        # if round(y) < mask_range[0] or round(y) >= mask_range[1]:
        #     continue
        # else:
        #     line_x.append(round(x))
        #     line_y.append(round(y))
        if (y, x) in nonzero_region:
            line_x.append(x)
            line_y.append(y)
    return line_x, line_y

def get_arc_points(center, theta_range=[0, 180], theta_step: int=1, radius=3):
    """centerを中心とする半円上の点のリストを返す

    Args:
        center (_type_): _description_
        theta_range (list, optional): _description_. Defaults to [0, 180].
        radius (int, optional): The radius of the arc. The larger, the more detailed the arc points will be. Defaults to 3.

    Returns:
        [x_list, y_list] (List[List[float], List[float]]): _description_
    """
    cx, cy = center
    theta = np.arange(theta_range[0], theta_range[1], step=theta_step)
    return [cx + radius*np.cos(np.deg2rad(theta)), cy - radius*np.sin(np.deg2rad(theta)), theta]

def get_gravity_center(attention_map):
    """return coordinate of gravity center of attention map in (x, y) order

    Args:
        attention_map (_type_): _description_

    Returns:
        _np.ndarray[int]: _description_
    """
    M = cv2.moments(attention_map)
    g = np.array([int(M["m10"] / (M["m00"] + 0.00001)), int(M["m01"] / (M["m00"] + 0.00001))])
    return g

def arg_max_crd(map):
    """return coordinate of max value in map in (x, y) order

    Args:
        map (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.unravel_index(np.argmax(map), map.shape)[::-1]

def gravity_center_1d(lst: list):
    """
    Args:
        lst (list): 1d list
    """
    M = sum(lst)
    moment = sum([i * lst[i] for i in range(len(lst))])
    return moment / M