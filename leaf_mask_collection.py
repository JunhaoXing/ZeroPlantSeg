import os
import glob
import argparse
import yaml
from tqdm import tqdm
import multiprocessing as mp

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

from OVSeg.open_vocab_seg import add_ovseg_config
from OVSeg.open_vocab_seg.utils import SAMVisualizationDemo
from utils.path import filename_wo_ext
from utils.fileio import str_to_tuple
from sliding_window.sliding_window import sliding_window, save_masks, save_masks_not_on_boundary

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def inference(img_paths: list, output_dir: str, config_path: str):
    
    mp.set_start_method("spawn", force=True)
    ovseg_config_file = './OVSeg/configs/ovseg_swinB_vitL_demo.yaml'
    ovseg_config = setup_cfg(ovseg_config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = config['leaf_mask_collection']
    class_names = config['class_names']
    class_names = class_names.split(',')
    trg_class_names = config['trg_class_names']
    trg_class_names = trg_class_names.split(',')
    granularity = config['granularity']
    min_mask_area = config['min_mask_area']
    
    
    demo = SAMVisualizationDemo(ovseg_config, granularity, './weights/sam_vit_h_4b8939.pth', './OVSeg/weights/ovseg_clip_l_9a1909.pth')
    for img_path in tqdm(img_paths):
        
        # load img
        img = read_image(img_path, format="BGR")
        
        
        # inference on raw img
        leaf_segs = []
        _, _, ins_segs = demo.run_on_image(img, class_names)        
        for cls_idx, ins_seg in enumerate(ins_segs):
            if class_names[cls_idx] in trg_class_names:
                for item in ins_seg:
                    item['win_id'] = 999 # id for not cropped (raw) img
                leaf_segs.extend(ins_seg)
        
        if config['use_sliding_window']:
            crop_imgs = sliding_window(img, str_to_tuple(config['dsize']))
            
            # inference on each cropped img
            for crop_id, crop_img in enumerate(crop_imgs):
                
                _, _, ins_segs = demo.run_on_image(crop_img, class_names)
                
                # extend leaf_segs with masks of each cropped img
                for cls_idx, ins_seg in enumerate(ins_segs):
                    if class_names[cls_idx] in trg_class_names:
                        for item in ins_seg:
                            item['win_id'] = crop_id
                        leaf_segs.extend(ins_seg)
            
            # save masks not on boundary
            print(leaf_segs)
            img_name = filename_wo_ext(img_path)
            if len(trg_class_names) == 1:
                save_folder = os.path.join(output_dir, img_name)
                os.makedirs(save_folder, exist_ok=True)
                save_masks_not_on_boundary(leaf_segs, save_folder, win_size=(512,512), min_area=min_mask_area)
            else:
                for cls_idx, trg_class_name in enumerate(trg_class_names):
                    save_folder = os.path.join(output_dir, img_name, trg_class_name)
                    os.makedirs(save_folder, exist_ok=True)
                    save_masks_not_on_boundary(leaf_segs[cls_idx], save_folder, win_size=(512,512), min_area=min_mask_area)
        else:
            # save all masks
            img_name = filename_wo_ext(img_path)
            save_folder = os.path.join(output_dir, img_name)
            os.makedirs(save_folder, exist_ok=True)
            save_masks(leaf_segs, save_folder, min_mask_area)
    
    
    return

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name", choices={'phenobench', 'growliflower', 'sb20'})
    parser.add_argument('--dataset_dir',type=str)
    parser.add_argument("--mode", type=str, help="dataset type", )
    parser.add_argument("--strip",type=str,default="*.png")
    parser.add_argument("--base_path",type=str, default="./output_p")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = setup_args()
    
    #img_dir = os.path.join('/datasets', args.dataset, args.mode, 'images')
    img_dir = args.dataset_dir
    img_paths = glob.glob(os.path.join(img_dir, args.strip))
    output_dir = os.path.join(args.base_path,'leaf_mask', args.dataset, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join('configs', args.dataset + '.yaml')
    
    inference(img_paths, output_dir, config_path)