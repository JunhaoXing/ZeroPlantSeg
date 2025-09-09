import cv2
import numpy as np


def crop_with_mask(rgb_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_cropped_img = rgb_img.copy()
    mask_cropped_img[mask == 0] = 0
    bbox = cv2.boundingRect(mask)
    x, y, w, h = bbox
    mask_cropped_img = mask_cropped_img[y:y+h, x:x+w]
    return mask_cropped_img

def resize2square(img: np.ndarray, length: int, interpolation=cv2.INTER_CUBIC, grayscale=False) -> np.ndarray:
    """一辺がlengthの正方形に収まる用にimgをリサイズする．

    Args:
        img (np.ndarray): RGB (or BGR) img
        length (int): length of output square

    Returns:
        np.ndarray: square img (length, length, 3)
    """
    if grayscale:
        output = np.zeros((length, length), dtype=np.uint8)
    else:
        output = np.zeros((length, length, 3), dtype=np.uint8)
    
    h, w = img.shape[:2]
    if h > w:
        res_img = cv2.resize(img, (int(length * w / h), length), interpolation=interpolation)
    else:
        res_img = cv2.resize(img, (length, int(length * h / w)), interpolation=interpolation)
    res_h, res_w = res_img.shape[:2]
    cy, cx = length / 2, length / 2
    output[int(cy - res_h / 2) : int(cy + res_h / 2), int(cx - res_w / 2) : int(cx + res_w / 2)] = res_img
    return output

def mask2bbox(mask: np.ndarray) -> tuple:
    """Convert mask to bbox (x, y, w, h)

    Args:
        mask (np.ndarray): _description_

    Returns:
        tuple: _description_
    """
    bbox = cv2.boundingRect(mask)
    return bbox

def crop_att_map_with_bbox(feature_map: np.ndarray):
    amplified = feature_map * 10000
    amplified = amplified.astype(np.uint8)
    bbox = mask2bbox(amplified)
    x, y, w, h = bbox
    cropped = feature_map[y:y+h, x:x+w]
    return cropped

def cvt2psimg(labels_img):
    """Convert labels image to uint8 gray scale image

    Args:
        labels_img (np.ndarray): labels image shape (h, w, 3)

    Returns:
        np.ndarray: gray scale image shape (h, w)
    """
    
    flatten_labels_img = labels_img.reshape(-1, 3)
    colors = np.unique(flatten_labels_img, axis=0)
    ps_img = np.zeros((labels_img.shape[0], labels_img.shape[1]), dtype=np.uint8)
    for mask_id, color in enumerate(colors):
        if color[0] == 0 and color[1] == 0 and color[2] == 0:
            continue
        else:
            # labels_imgのcolorに対応する部分のみを抜き出す
            mask = np.where((labels_img == color).all(axis=2), 1, 0)
            ps_img[mask == 1] = mask_id + 1
    return ps_img