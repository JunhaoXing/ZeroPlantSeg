import os
import re
import glob
import argparse
import sys
sys.path.append("../")

import numpy as np
import cv2
from tqdm import tqdm

from metrics import PQ
from utils.fileio import load_image_as_tensor, save_dict_list_as_csv

DATASET_DIR = "../../dataset/"
OUTPUT_DIR = "../output/"

def main(args):
    pred_dir = args.pred_dir
    save_path = args.save_path
    
    gt_ps_img_dir = args.gt_dir
    pred_ps_img_paths = glob.glob(os.path.join(pred_dir, "*.png"))

    eval_list = []
    for pred_ps_img_path in tqdm(pred_ps_img_paths, ascii=True):
        filename = os.path.basename(pred_ps_img_path)
        gt_ps_img_path = os.path.join(gt_ps_img_dir, filename)
        
        pred_ps_img = cv2.imread(pred_ps_img_path, -1)
        gt_ps_img = cv2.imread(gt_ps_img_path, -1)
        
        pq, sq, rq = PQ(gt_ps_img, pred_ps_img)
        
        eval_list.append({"filename": filename, "pq": pq, "sq": sq, "rq": rq})

    pq_list = [e["pq"] for e in eval_list if e["pq"] is not None]
    sq_list = [e["sq"] for e in eval_list if e["sq"] is not None]
    rq_list = [e["rq"] for e in eval_list if e["rq"] is not None]

    print("pred_dir: {}".format(pred_dir))
    print("PQ: {:.4f}".format(np.mean(pq_list)))
    print("SQ: {:.4f}".format(np.mean(sq_list)))
    print("RQ: {:.4f}".format(np.mean(rq_list)))
    
    if save_path != None:
        save_dict_list_as_csv(os.path.join(save_path), eval_list)
    
    return


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=["PhenoBench", "GrowliFlower", "SB20"])
    parser.add_argument('--mode', required=True, choices=["val", "test"])
    parser.add_argument('--task', required=True, choices=["plant", "leaf"])
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args()
    
    if args.task == "plant":
        task = "plant_instances"
    elif args.task == "leaf":
        task = "leaf_instances"
    
    args.pred_dir = os.path.join(OUTPUT_DIR, args.dataset, args.mode, task)
    args.gt_dir = os.path.join(DATASET_DIR, args.dataset, args.mode, task)
    
    if os.path.exists(args.pred_dir) == False:
        print(f"pred_dir: {args.pred_dir} does not exist.")
        sys.exit(1)
    if os.path.exists(args.gt_dir) == False:
        print(f"gt_dir: {args.gt_dir} does not exist.")
        sys.exit(1)
    
    main(args)