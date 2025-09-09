import os
from typing import List, Dict, Any
import csv

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

def read_csv_as_dict_list(csv_path: str) -> List[Dict]:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dict_list = [row for row in reader]
    return dict_list

def save_dict_list_as_csv(save_path: str, dict_list: List[Dict]) -> None:

    if len(dict_list) == 0:
        return
    
    labels = list(dict_list[0].keys())
    
    with open(save_path,'w', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=labels)
        writer.writeheader()
        writer.writerows(dict_list)
    
    return

def cvt_strcrd(str_crd: str, mode='int') -> List[int]:
    """convert string coordinate like '(12, 13)' or '[13, 14]' to list of int

    Args:
        str_crd (str): _description_
        mode (str, optional): _description_. Defaults to 'float'.

    Returns:
        List[int]: _description_
    """
    int_crd_list = [int(num) for num in str_crd.strip('()[]').split(',')]
    
    if mode == 'float':
        return list(map(float, int_crd_list))
    
    return int_crd_list

def write_masks_to_folder(masks, predicted_ious=None, save_dir: str=None) -> None:
    if predicted_ious is None:
        for i, mask in enumerate(masks):
            mask = mask[0].cpu().numpy()
            cv2.imwrite(os.path.join(save_dir, str(i) + ".png"), mask.astype(np.uint8) * 255)
    else:
        predicted_iou_dict_list = []
        for i, data in enumerate(zip(masks, predicted_ious)):
            mask, predicted_iou = data
            mask = mask[0].cpu().numpy()
            cv2.imwrite(os.path.join(save_dir, str(i) + ".png"), mask.astype(np.uint8) * 255)
            predicted_iou_dict_list.append({"mask_id": i, "predicted_iou": predicted_iou.item()})
        save_dict_list_as_csv(os.path.join(save_dir, "predicted_ious.csv"), predicted_iou_dict_list)

    return

def write_masks_w_metadata_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def load_image_as_tensor(path_to_file: str) -> torch.Tensor:
    to_tensor = T.ToTensor()
    img_tensor = Image.open(path_to_file)
    img_tensor = to_tensor(img_tensor)
    img_tensor = img_tensor.squeeze() # [H x W]
    return img_tensor

def read_stem_estim_result(csv_path):
    if os.path.isfile(csv_path) is False:
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    result_data = read_csv_as_dict_list(csv_path)
    # load data
    ref_crds, root_crds, tip_crds, mask_ids = [], [], [], []
    for result_datum in result_data:
        ref_crds.append(tuple(map(int, result_datum["ref_crd"][1:-1].split(', '))))
        root_crds.append(tuple(map(int, result_datum["root_crd"][1:-1].split(', '))))
        if result_datum["tip_crd"] == "None":
            tip_crds.append(root_crds[-1])
        else:
            tip_crds.append(tuple(map(int, result_datum["tip_crd"][1:-1].split(', '))))
        mask_ids.append(int(result_datum["id"]))
    
    return ref_crds, root_crds, tip_crds, mask_ids

def str_to_tuple(str_crd: str) -> tuple:
    return tuple(map(int, str_crd.strip('()').split(',')))

if __name__ == "__main__":
    temp = [{"a": 0, "b": 1}, {"a": 2, "b": 3}]
    save_dict_list_as_csv("temp.csv", temp)
