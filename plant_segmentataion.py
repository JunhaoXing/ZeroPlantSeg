import os
import glob
import yaml
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from tqdm import tqdm

from utils.path import basename, filename_wo_ext
from utils.fileio import str_to_tuple
from attention_map.gdino_attention import load_groundingDINO
from attention_map.get_leaf_root import get_leaf_root_wls

def load_binary_masks_as_dict(masks_dir) -> Dict[int, np.ndarray]:
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
    masks = [cv2.imread(mask_path, -1) for mask_path in mask_paths]
    mask_ids = [int(filename_wo_ext(mask_path)) for mask_path in mask_paths]
    
    return {mask_id: mask for mask_id, mask in zip(mask_ids, masks)}

def greedyDBSCANclustering(clustering_points, eps: float, min_samples: int, num_steps: int=10):
    
    # init
    clustering_result = np.array([-1] * len(clustering_points))
    unclustered_idx_list = np.array(list(range(len(clustering_points)))) # index of unclustered points
    unclustered_points = np.array(clustering_points.copy())
    
    for i in range(num_steps):
        # clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(unclustered_points)
        labels = clustering.labels_
        
        # get largest cluster
        outlier_removed_labels = labels[np.where(labels!=-1)]
        if len(outlier_removed_labels) == 0: # if no more clusters are formed, exit
            break
        else:
            counter = Counter(outlier_removed_labels)
            largest_cluster_label = counter.most_common(1)[0][0]
            
            labels = np.array(labels)
            largest_cluster_idx_list = np.where(labels==largest_cluster_label)
            largest_cluster_orgn_idx_list = unclustered_idx_list[largest_cluster_idx_list]
            clustering_result[largest_cluster_orgn_idx_list] = i+1
            
            # remove clustered points in the largest cluster
            unclustered_idx_list = np.delete(unclustered_idx_list, largest_cluster_idx_list)
            ## reform unclustered_points
            unclustered_points = np.array(clustering_points.copy())[unclustered_idx_list]
            
            if unclustered_points.shape[0] == 0: # if the all the points are clustered, exit
                break
    
    return clustering_result

def majority_voting_masks_plants(leaf_masks_dict, clustering, mask_ids, 
                                 outlier_label=0, 
                                 outlier_color=9999, 
                                 output_img_size=(1024,1024), 
                                 min_area_th=0, 
                                 max_area_th=1024*1024) -> np.ndarray:
    
    plant_seg = np.zeros(output_img_size).astype(np.uint16)
    H, W = output_img_size
    voting_list = [[[] for w in range(W)] for h in range(H)]
    
    # make voting list
    for clustering_label, mask_id in zip(clustering, mask_ids):
        mask = leaf_masks_dict[mask_id]
        if np.count_nonzero(mask) > max_area_th or np.count_nonzero(mask) < min_area_th:
            continue
        
        if clustering_label == outlier_label:
            mask_ys, mask_xs = np.where(mask > 0)
            for y, x in zip(mask_ys, mask_xs):
                voting_list[y][x].append(outlier_color)
        else:
            mask_ys, mask_xs = np.where(mask > 0)
            for y, x in zip(mask_ys, mask_xs):
                voting_list[y][x].append(clustering_label)
    
    # make plant_seg
    for h in range(H):
        for w in range(W):
            if len(voting_list[h][w]) > 0:
                plant_seg[h, w] = max(voting_list[h][w], key=voting_list[h][w].count)
    
    return plant_seg

def min_MHLdist_label(plant_seg, clustering_points, outlier_crd, clustering, outlier_label=0, dist_th=64):
    
    covmat_dict = {}
    clustering_points_arr = np.array(clustering_points)
    for label in np.unique(clustering):
        crds = clustering_points_arr[np.where(clustering == label)]
        covmat = np.cov(crds, rowvar=False)
        covmat_dict[label] = covmat
    
    min_dist = float('inf')
    min_dist_label = outlier_label
    for label in np.unique(clustering):
        if label == outlier_label:
            continue
        crnt_plant_mask = plant_seg == label
        contours = cv2.findContours(crnt_plant_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            for pt in contour:
                dist = distance.mahalanobis(pt[0], outlier_crd, covmat_dict[label])
                if dist > 0 and dist < min_dist and dist < dist_th:
                    min_dist = dist
                    min_dist_label = label
    
    return min_dist_label

def post_process_outliers(plant_seg: np.ndarray, 
                          clustering_points: np.ndarray, 
                          clustering, 
                          outlier_label=0, 
                          dist_th=64,
                          GENERATE_NEW_CLUSTER=False):
    
    outlier_idxes = [idx for idx, label in enumerate(clustering) if label == outlier_label]
    processed_clustering = clustering.copy()
    
    new_cluster_id = processed_clustering.max() + 1
    for outlier_idx in outlier_idxes:
        outlier_crd = clustering_points[outlier_idx]
        revised_label = min_MHLdist_label(plant_seg, clustering_points, outlier_crd, clustering, outlier_label, dist_th=dist_th)
        
        if revised_label != outlier_label:
            processed_clustering[outlier_idx] = revised_label
        elif GENERATE_NEW_CLUSTER:
            processed_clustering[outlier_idx] = new_cluster_id
            new_cluster_id += 1
    
    return processed_clustering

def plant_segmentation(bgr_img_dir, masks_dirs: list, config_path, output_dir, GENERATE_NEW_CLUSTER=True, VALIDATION_MODE=False):
    
    if VALIDATION_MODE:
        masks_dirs = masks_dirs[:1]
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = config['plant_segmentation']
    eps_minsamples_dict = config['eps_minsamples_dict']
    num_steps = config['num_steps']
    min_area_th = config['min_area_th']
    output_img_size = str_to_tuple(config['output_img_size'])
    
    grounding_dino_model = load_groundingDINO()
    
    for masks_dir in tqdm(masks_dirs):
        img_name = basename(masks_dir)
        img_filename = img_name + ".png"
        bgr_img_path = os.path.join(bgr_img_dir, img_filename)
        img_date = next((date for date in eps_minsamples_dict.keys() if date in img_name), None)
        eps, min_samples = str_to_tuple(eps_minsamples_dict[img_date])
        
        # calc leaf keypoints
        clustering_points, mask_ids = get_leaf_root_wls(bgr_img_path, masks_dir, eps, grounding_dino_model)

        # clustering
        if len(clustering_points) > 0:
            clustering = greedyDBSCANclustering(clustering_points, eps=eps, min_samples=min_samples, num_steps=num_steps)

            # add label offset
            clustering = clustering + 1
            outlier_label = 0
            
            # make plant segmentation output without post processing
            leaf_mask_dict = load_binary_masks_as_dict(masks_dir)
            plant_seg_wo_pp = majority_voting_masks_plants(leaf_mask_dict, clustering, mask_ids, outlier_label=outlier_label, output_img_size=output_img_size, min_area_th=min_area_th)

            # make plant segmentation output with post processing
            processed_clustering = post_process_outliers(plant_seg_wo_pp, clustering_points, clustering, outlier_label, GENERATE_NEW_CLUSTER=GENERATE_NEW_CLUSTER)
            plant_seg_w_pp = majority_voting_masks_plants(leaf_mask_dict, processed_clustering, mask_ids, outlier_label=outlier_label, output_img_size=output_img_size, min_area_th=min_area_th)
        else:
            # make zeros output
            plant_seg_wo_pp = np.zeros(output_img_size).astype(np.uint16)
            plant_seg_w_pp = np.zeros(output_img_size).astype(np.uint16)
        
        # save
        if not VALIDATION_MODE:
            cv2.imwrite(os.path.join(output_dir, img_filename), plant_seg_w_pp)
    
    return plant_seg_wo_pp, plant_seg_w_pp

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20'})
    parser.add_argument("--mode", type=str, help="dataset type", choices={'val', 'test'})
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = setup_args()
    bgr_img_dir = os.path.join('dataset', args.dataset, args.mode, 'images')
    masks_root_dir = os.path.join('output/leaf_mask', args.dataset, args.mode)
    masks_dirs = glob.glob(os.path.join(masks_root_dir, "*/"))
    output_dir = os.path.join('output/plant_instance', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=False)
    config_path = os.path.join('configs', args.dataset + '.yaml')
    
    plant_segmentation(bgr_img_dir, masks_dirs, config_path, output_dir=output_dir, GENERATE_NEW_CLUSTER=True)