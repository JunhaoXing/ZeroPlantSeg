import os
import glob
import argparse
from typing import List

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from leafonlysam.evaluation import COCOEvaluator

def checkcolour(masks, hsv):
    colours = np.zeros((0,3))

    for i in range(len(masks)):
        color = hsv[masks[i]['segmentation']].mean(axis=(0))
        colours = np.append(colours,color[None,:], axis=0)
        
    idx_green = (colours[:,0]<75) & (colours[:,0]>35) & (colours[:,1]>35)
    if idx_green.sum()==0:
        # grow lights on adjust
        idx_green = (colours[:,0]<100) & (colours[:,0]>35) & (colours[:,1]>35)
    
    return(idx_green)

def checkfullplant(masks):
    mask_all = np.zeros(masks[0]['segmentation'].shape[:2])

    for mask in masks:
        mask_all +=mask['segmentation']*1
        
    iou_withall = []
    for mask in masks:
        iou_withall.append(iou(mask['segmentation'], mask_all>0))
        
    idx_notall = np.array(iou_withall)<0.9
    return idx_notall

def getbiggestcontour(contours):
    nopoints = [len(cnt) for cnt in contours]
    return(np.argmax(nopoints))

def checkshape(masks):
    cratio = []

    for i in range(len(masks)):
        test_mask = masks[i]['segmentation']
        
        if not test_mask.max():
            cratio.append(0)
        else:

            contours,hierarchy = cv2.findContours((test_mask*255).astype('uint8'), 1, 2)

            # multiple objects possibly detected. Find contour with most points on it and just use that as object
            cnt = contours[getbiggestcontour(contours)]
            M = cv2.moments(cnt)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)

            (x,y),radius = cv2.minEnclosingCircle(cnt)

            carea = np.pi*radius**2

            cratio.append(area/carea)
    idx_shape = np.array(cratio)>0.1
    return(idx_shape)

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def get_masks_NotAll(masks: List[np.ndarray], th_iou: float=0.9) -> List[np.ndarray]:
    if len(masks) == 0:
        return masks, []
    else:
        masks_NotAll = []
        mask_union = np.zeros_like(masks[0])
        for mask in masks:
            mask_union[np.where(mask > 0)] = 255
        
        chosen_masks_idx = []
        for i, mask in enumerate(masks):
            if iou(mask, mask_union) < th_iou:
                masks_NotAll.append(mask)
                chosen_masks_idx.append(i)
        
        
        return masks_NotAll, chosen_masks_idx

def issubset(mask1, mask2, th_ratio=0.9):
    # is mask2 subpart of mask1
    intersection = np.logical_and(mask1, mask2)
    return(np.sum(intersection)/np.count_nonzero(mask2)>th_ratio)

def istoobig(masks: List[np.ndarray], th_ratio=0.9, return_mask_all=False) -> List[int]:
    idx_toobig = []
    
    mask_all = np.zeros_like(masks[0]) # the number of masks that cover each pixel

    for mask in masks:
        mask_all[np.where(mask > 0)] += 1 

    for idx in range(len(masks)):
        if idx in idx_toobig:
            continue
        for idx2 in range(len(masks)):
            if idx==idx2:
                continue
            if idx2 in idx_toobig:
                continue
            # check if masks[idx2] is 90% contained in masks[idx]
            if issubset(masks[idx2], masks[idx], th_ratio): 
                # check if actually got both big and small copy delete if do
                if mask_all[np.where(masks[idx2] > 0)].mean() > 1.5:
                    idx_toobig.append(idx2)
    
    idx_toobig.sort(reverse=True)
    if return_mask_all:
        return idx_toobig, mask_all
    return (idx_toobig)

def remove_toobig(masks: List[np.ndarray], idx_toobig, th_ratio=0.9, return_idx_del=False):
    masks_ntb = masks.copy()

    idx_del = []
    for idxbig in idx_toobig[1:]: # why 1:?
        maskbig = masks_ntb[idxbig].copy()
        submasks = np.zeros(maskbig.shape)

        for idx in range(len(masks_ntb)):
            if idx==idxbig:
                continue
            if issubset(masks_ntb[idxbig], masks_ntb[idx], th_ratio):
                submasks +=masks_ntb[idx]

        crnt_ratio = np.count_nonzero(np.logical_and(maskbig, submasks>0)) / np.count_nonzero(maskbig)
        if crnt_ratio > th_ratio:
            # can safely remove maskbig
            idx_del.append(idxbig)
            del(masks_ntb[idxbig])
    
    if return_idx_del:
        return masks_ntb, idx_del
    
    return masks_ntb

def process_leaf_only_sam(model, image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # use crop_n_layer=1 to improve results on smallest leaves 
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,  
    )

    # get masks
    masks = mask_generator.generate(image)
    
    # remove things that aren't green enough to be leaves
    idx_green = checkcolour(masks,hsv)

    masks_g = []
    for idx, use in enumerate(idx_green):
        if use:
            masks_g.append(masks[idx])

    if len(masks_g) > 2:

        # check to see if full plant detected and remove
        idx_notall = checkfullplant(masks_g)

        masks_na = []

        for idx, use in enumerate(idx_notall):
            if use:
                masks_na.append(masks_g[idx])

    else:
        masks_na = masks_g

    idx_shape = checkshape(masks_na)

    masks_s = []
    for idx, use in enumerate(idx_shape):
        if use:
            masks_s.append(masks_na[idx])

    idx_toobig = istoobig(masks_s)
    masks_ntb = remove_toobig(masks_s, idx_toobig)
    
    return masks_ntb


def eval(args, img_dir, res_dir, evaluator=None):
    
    sam_checkpoint = args.sam_ckpt
    model_type = args.model_type
    device = args.device
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    inputs = [{'image_id': 1000+i} for i in range(len(img_paths))]
    outputs = []
    for img_path in tqdm(img_paths):
        imname = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        v = Visualizer(image[:, :, ::-1], scale=1, instance_mode=ColorMode.SEGMENTATION)
        
        lsam_pred_masks = process_leaf_only_sam(sam, image)
        
        pred_boxes = Boxes([mask["bbox"] for mask in lsam_pred_masks])
        pred_masks = torch.tensor(np.array([mask["segmentation"].astype(int) for mask in lsam_pred_masks])) # Convert bool to int
        pred_scores = np.array([mask["predicted_iou"] for mask in lsam_pred_masks])
        pred_classes = torch.ones(pred_masks.shape[0])

        instances = Instances(image_size=(image.shape[0], image.shape[1]),
                              pred_boxes=pred_boxes,
                              scores=torch.tensor(pred_scores),
                              pred_classes=pred_classes,
                              pred_masks=pred_masks)
        
        output = {'instances': instances}
        outputs.append(output)
        
        mask = {"isntances": Instances(image_size=(image.shape[0], image.shape[1]), pred_masks=pred_masks)}
        out = v.draw_instance_predictions(mask["isntances"])
        res_img = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(res_dir, imname), res_img)
        
        # # save results at each step as npz file 
        # np.savez(folder_out + imname.replace('.JPG','leafonly_allmasks.npz'),
        #           masks, masks_g, masks_na, masks_s, masks_ntb)

    if evaluator is not None:
        evaluator.process(inputs, outputs)
        evaluator.evaluate()
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_ckpt', default='../weights/sam_vit_h_4b8939.pth')
    parser.add_argument('--model_type', default='vit_h')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--species', type=str)
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--only_inference', action='store_true')
    args = parser.parse_args()
    
    img_dir = f'./data/segmentation/p_{args.species}/test/img'
    res_dir = os.path.join(args.out_dir, f'./results_{args.species}')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    if not args.only_inference:
        register_coco_instances('leaf_only_sam', {}, f'/data/segmentation/p_{args.species}/{args.species}_test.json', args.img_dir)
        evaluator = COCOEvaluator('leaf_only_sam', output_dir=args.res_dir)
        evaluator.reset()
    
    eval(args, img_dir, res_dir)#, evaluator)
