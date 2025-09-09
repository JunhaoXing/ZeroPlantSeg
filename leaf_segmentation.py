import os
import glob
import yaml
import argparse
import csv
import cv2
import numpy as np
from tqdm import tqdm

from utils.path import basename
from utils.fileio import str_to_tuple
from evaluation.metrics import area
from leafonlysam.LeafOnlySAM import istoobig, remove_toobig


def merge_leaf_masks(masks_dir: str, min_area_th: float=0, max_area_th: float=np.inf, th_ratio=0.9, output_img_size =(1024, 1024)):
    
    # load metadata
    metadata_filename = 'metadata.csv'
    csv_path = os.path.join(masks_dir, metadata_filename)
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        metadata = [data for data in reader]
    metadata = sorted(metadata, key=lambda x: int(x['area']), reverse=True) # sort by area in descending order
    
    # load mask images
    masks = []
    for mask_data in metadata:
        # load mask
        try:
            mask_filename = mask_data['id'] + '.png'
        except:
            mask_filename = mask_data['filename']
        mask_path = os.path.join(masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    
    # integration
    idx_toobig = istoobig(masks, th_ratio)
    masks_not_duplicate = remove_toobig(masks, idx_toobig, th_ratio)
    
    # make output
    output_img = np.zeros(output_img_size).astype(np.uint16) # init output img
    for i, mask in enumerate(masks_not_duplicate):
        if area(mask) > max_area_th or area(mask) < min_area_th:
            continue
        output_img[np.where(mask != 0)] = i+1
    
    return output_img

def make_leaf_seg(masks_dirs: list, output_dir: str, config, VALIDATION_MODE=False):
    
    if not VALIDATION_MODE:
        os.makedirs(output_dir, exist_ok=True)
    else:
        outputs = []
    
    # load parameters
    with open(config) as f:
        config = yaml.safe_load(f)
        config = config['leaf_segmentation']
    min_area_th = config['min_area_th']
    max_area_th = config['max_area_th']
    th_ratio = config['th_ratio']
    output_img_size = str_to_tuple(config['output_img_size'])
    
    # integration
    for masks_dir in tqdm(masks_dirs):
        leaf_segmentation_output = merge_leaf_masks(masks_dir, min_area_th=min_area_th, max_area_th=max_area_th, th_ratio=th_ratio, output_img_size=output_img_size)
        img_name = basename(masks_dir) + '.png'
        # output
        if not VALIDATION_MODE:
            cv2.imwrite(os.path.join(output_dir, img_name), leaf_segmentation_output)
        else:
            outputs.append(leaf_segmentation_output)
    
    # output settings.txt
    if VALIDATION_MODE:
        return outputs
    else:
        # with open(os.path.join(output_dir, "settings.txt"), mode='w') as f:
        #     f.write("th_IoU: " + str(th_IoU) + "\n")
        #     f.write("masks_metrics: " + masks_metrics + "\n")
        #     f.write("min_area_th: " + str(min_area_th) + "\n")
        #     f.write("max_area_th: " + str(max_area_th) + "\n")
        return

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20'})

    parser.add_argument("--mode", type=str, help="dataset type", )
    parser.add_argument("--strip",type=str,default="*.png")
    parser.add_argument("--base_path",type=str,default="./output_p")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = setup_args()
    base_path = args.base_path
    masks_root_dir = os.path.join(base_path,'leaf_mask', args.dataset, args.mode)
    masks_dirs = glob.glob(os.path.join(masks_root_dir, "*/"))
    output_dir = os.path.join(base_path,'leaf_instance', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join('configs', args.dataset + '.yaml')
    
    make_leaf_seg(masks_dirs, output_dir, config_path)