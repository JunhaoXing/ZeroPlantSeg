import os
import glob
import random
import argparse
from typing import List

import numpy as np
import cv2
from groundingdino.util.inference import predict

from attention_map.edit_mask import crop_with_mask, resize2square, mask2bbox
from attention_map.gdino_attention import load_groundingDINO, get_attention_map, cvt_img2tensor, saveAttentionMapMetadata_list
from attention_map.gdino_attention import AttentionMapMetaData
from attention_map.direction_map import get_line_points, arg_max_crd, gravity_center_1d
from leafonlysam.LeafOnlySAM import get_masks_NotAll
from utils.path import filename_wo_ext, basename

def load_binary_masks(masks_dir):
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
    masks = [cv2.imread(mask_path, -1) for mask_path in mask_paths]
    mask_ids = [int(filename_wo_ext(mask_path)) for mask_path in mask_paths]
    return masks, mask_ids

def wls(featmap: np.ndarray):
    # WLS
    b, a = np.nonzero(featmap) # (1, N), (1, N)
    c = featmap[b, a] # (N,)

    A = np.array([[1, a_k] for a_k in a]) # (N, 2)
    b = b.transpose() # (N, 1)
    C = np.diag(c) # (N, N)

    try:
        i, s = np.linalg.inv(A.T @ C @ A) @ A.T @ C @ b
    except:
        i, s = None, None
    
    return i, s # y-intercept, slope

def calc_leaf_keypoints(rgb_img: np.ndarray, mask: np.ndarray, feat_map_metadata: AttentionMapMetaData, 
                                     detection_model, mask_square_length:int=512, text_prompt:str="stem"):
    
    cropped_img = crop_with_mask(rgb_img, mask)
    cropped_img = resize2square(cropped_img, mask_square_length)
    
    image_tensor = cvt_img2tensor(cropped_img)
    predict(
        model=detection_model, 
        image=image_tensor,
        caption=text_prompt, 
        box_threshold=0.3, 
        text_threshold=0.3
    )
    one_feature_map = get_attention_map(cropped_img)
    
    """
    plt.imshow(one_feature_map)
    plt.show()
    """
    
    # get stem line
    center = arg_max_crd(one_feature_map)
    stem_i, stem_s = wls(one_feature_map) # y-intercept, slope of stem line
    if stem_i is None or stem_s is None:
        return # None?
    line_theta = np.rad2deg(np.arctan(stem_s))
    line_x, line_y = get_line_points(one_feature_map, a=stem_s, b=stem_i)
    
    
    # rotate feature map by -line_theta in the window size of 2r x 2r
    ## get affine matrix
    h, w = one_feature_map.shape
    pts = np.array([(0,0), (w,0), (w,h), (0,h)])
    ctr = np.array(center)
    r = np.sqrt(max(np.sum((pts-ctr)**2, axis=1)))
    win_H, win_W = round(2*r), round(2*r)
    
    affine_mat = cv2.getRotationMatrix2D(center=(round(center[0]), round(center[1])), angle=-line_theta, scale=1) # shape (2, 3)
    affine_mat[0][2] += r - center[0]
    affine_mat[1][2] += r - center[1]
    
    ## warp
    rotated_feature_map = cv2.warpAffine(one_feature_map, affine_mat, dsize=(win_W, win_H))
    
    # calc average for each column 
    column_means = []
    for j in range(rotated_feature_map.shape[1]):
        if np.max(rotated_feature_map[:, j]) > 0:
            nonzero_idx_column = np.where(rotated_feature_map[:, j] != 0)
            column_mean = np.mean(rotated_feature_map[nonzero_idx_column, j])
            column_means.append(column_mean)
        else:
            column_means.append(0)
    
    """
    plt.imshow(rotated_feature_map)
    plt.colorbar()
    plt.show()
    
    plt.imshow(mean_feature_map)
    plt.colorbar()
    plt.show()
    """
    
    # get gravity center of processed feature map
    gx = gravity_center_1d(column_means)
    gy = win_H / 2
    
    # inverse rotate gravity center to feature map coordinate
    gc = np.array([gx, gy, 1])
    affine_mat = np.insert(affine_mat, 2, np.array([0, 0, 1]), axis=0) # shape (2, 3) -> (3, 3)
    inv_affine_mat = np.linalg.inv(affine_mat)
    gc = inv_affine_mat @ gc # to feature map coordinate (同次座標)
    
    # save reference point in input image coordinate
    ref_point = feat_map_metadata.cvt2orgn_crd(gc[:2]) # to original image coordinate
    feat_map_metadata.ref_point = ref_point
    
    # get the nearest end point of stem line from the gravity center
    if len(line_x) == 0 or len(line_y) == 0: # if the stem line is not detected
        feat_map_metadata.leaf_tip_crd = ref_point
        return ref_point # return reference point as leaf root
    else:
        end_pts = ((line_x[0], line_y[0]), (line_x[-1], line_y[-1])) # feature map coordinate
        dists = [np.linalg.norm(np.array([x, y]) - gc[:2]) for x, y in end_pts]
        min_dist_idx = np.argmin(dists)
        max_dist_idx = np.argmax(dists)
        leaf_root_crd = feat_map_metadata.cvt2orgn_crd(end_pts[min_dist_idx]) # to original image coordinate
        leaf_tip_crd = feat_map_metadata.cvt2orgn_crd(end_pts[max_dist_idx]) # to original image coordinate
        return leaf_root_crd, leaf_tip_crd

def num_in_eps(all_points, target_point, eps):
    num = 0
    for point in all_points:
        if np.linalg.norm(np.array(point) - np.array(target_point)) < eps:
            num += 1
    return num

def get_leaf_root_wls(bgr_img_path, leaf_masks_dir, eps, grounding_dino_model):
    # load
    bgr_img = cv2.imread(bgr_img_path)
    masks, mask_ids = load_binary_masks(leaf_masks_dir)
    masks, chosen_masks_idx = get_masks_NotAll(masks)
    mask_ids = [mask_ids[i] for i in chosen_masks_idx]
    
    # calculation
    leaf_root_crds = []
    leaf_tip_crds = []
    for mask_id, mask in zip(mask_ids, masks):
        orgn_pos = mask2bbox(mask)
        metadata = AttentionMapMetaData(orgn_pos, length=512, id=mask_id)
        
        leaf_root_crd, leaf_tip_crd = calc_leaf_keypoints(bgr_img, mask, metadata, grounding_dino_model)
        
        leaf_root_crds.append(leaf_root_crd)
        leaf_tip_crds.append(leaf_tip_crd)
    
    # post processing
    all_points = leaf_root_crds + leaf_tip_crds
    rev_root_crds, rev_tip_crds = [], []
    for root_crd, tip_crd in zip(leaf_root_crds, leaf_tip_crds):
        root_num_in_eps = num_in_eps(all_points, root_crd, eps)
        tip_num_in_eps = num_in_eps(all_points, tip_crd, eps)
        if root_num_in_eps < tip_num_in_eps:
            rev_root_crds.append(tip_crd)
            rev_tip_crds.append(root_crd)
        else:
            rev_root_crds.append(root_crd)
            rev_tip_crds.append(tip_crd)
    
    return leaf_root_crds, mask_ids

def run_on_separate_masks():
    # setup
    # rgb_img_dir = "../dataset/PhenoBench/val/images/"
    # plant_ps_img_dir = "../dataset/PhenoBench/val/plant_instances/" # only for evaluation
    # leaf_masks_dirs = glob.glob("../OVSeg/output/20240516/*/")
    # save_dir = "output/stem_line_estimation/leaf_root_ovseg0516"
    # SAVE_EVAL = False
    
    rgb_img_dir = "../dataset/mySugarBeet2016/val/images/"
    plant_ps_img_dir = ""
    leaf_masks_dirs = glob.glob("../OVSeg/output/20240605_sugarbeet_val/*/")
    save_dir = "output/stem_line_estimation/leaf_root_ovseg0605_val"
    
    os.makedirs(save_dir, exist_ok=True)
    detection_model = load_groundingDINO()
    
    for i, leaf_masks_dir in enumerate(leaf_masks_dirs):
        print("Processing {}/{} image...".format(i+1, len(leaf_masks_dirs)))
        
        img_filename = basename(leaf_masks_dir) + '.png'
        rgb_img_path = os.path.join(rgb_img_dir, img_filename)
        rgb_img = cv2.imread(rgb_img_path)
        
        # load leaf masks
        masks, mask_ids = load_binary_masks(os.path.join(leaf_masks_dir)) #, 'green leaf')) # in the case for output of a specific class
        masks, chosen_masks_idx = get_masks_NotAll(masks) # remove masks that cover almost the entire image, using the method of LSAM
        mask_ids = [mask_ids[i] for i in chosen_masks_idx]
        
        # calculation
        leaf_masks_data = []
        for mask_id, mask in zip(mask_ids, masks):
            orgn_pos = mask2bbox(mask)
            metadata = AttentionMapMetaData(orgn_pos, length=512, id=mask_id)
            
            leaf_root_crd, leaf_tip_crd = calc_leaf_keypoints(rgb_img, mask, metadata, detection_model)
            
            metadata.leaf_root_crd = leaf_root_crd
            metadata.leaf_tip_crd = leaf_tip_crd
            leaf_masks_data.append(metadata)
        
        # save results
        img_save_dir = os.path.join(save_dir, os.path.splitext(img_filename)[0])
        os.makedirs(img_save_dir, exist_ok=True)
        
        saveAttentionMapMetadata_list(os.path.join(img_save_dir, "result.csv"), leaf_masks_data)
    
    return


if __name__ == "__main__":
    run_on_separate_masks()