import os
import glob
import cv2
import numpy as np
from utils.color import random_color

def visualize_separate_masks(bgr_img: np.ndarray, masks_dir: str, reverse=True, min_area=0) -> np.ndarray:
    # load masks
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))        
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, -1)
        masks.append(mask)

    # sort by area
    masks = sorted(masks, key=lambda x: np.count_nonzero(x), reverse=reverse)
    
    # draw masks
    drawn = bgr_img.copy()
    for mask in masks:
        if np.count_nonzero(mask) < min_area:
            continue
        drawn[mask > 0] = random_color()
    
    return drawn

def visualize_ps(bgr_img, ps_img, outlier_id: int=None, VIS_OUTLIERS: bool=False):
    drawn = bgr_img.copy()
    flatten_img = ps_img.reshape(-1)
    colors = sorted(np.unique(flatten_img, axis=0))

    for idx, clr in enumerate(colors):
        if clr != 0:
            mask_img = np.where(ps_img==clr, 1, 0)
            
            if VIS_OUTLIERS == False and clr == outlier_id:
                continue
            elif VIS_OUTLIERS == True and outlier_id is not None and clr == outlier_id:
                paint_color = (0, 0, 255)
            else:
                paint_color = random_color()
            
            drawn[mask_img>0] = paint_color
    
    return drawn