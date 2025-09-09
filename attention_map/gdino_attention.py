from typing import List

from PIL import Image
import numpy as np
import cv2
import torch

# from groundingdino.util.inference import load_model
# import groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import load_model
import GroundingDINO.groundingdino.datasets.transforms as T
from utils.fileio import save_dict_list_as_csv

def load_groundingDINO(config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", weights_path="GroundingDINO/weights/groundingdino_swint_ogc.pth"):
    detection_model = load_model(config_path, weights_path)
    return detection_model

def get_attention_map(mask: np.ndarray, attention_path: str="output/temp/text2img_attention.pt", query_idx: int=0) -> np.ndarray:
    """get attention map from text to image

    Args:
        mask (np.ndarray): will be converted to grayscale and resized to 100x100(featmap_ress[0])
        attention_path (str, optional): _description_. Defaults to "text2img_attention.pt".
        query_idx: index of query (text)

    Returns:
        np.ndarray : _description_
    """
    featmap_ress = [(100, 100), (50, 50), (25, 25), (13, 13)] # resolution of each image feature
    
    attention = torch.load(attention_path).cpu()
    attention_heads = attention[:, query_idx, :]

    head_feat_maps = []
    for attention_head in attention_heads:
        feat_maps = [] # feature map from same layer
        last_idx = 0
        for res in featmap_ress:
            # unflatten each resolution feature map
            first_idx = last_idx
            last_idx += res[0] * res[1]
            feature_map = attention_head[first_idx:last_idx]
            feature_map = feature_map.reshape(res[0], res[1])
            
            feature_map = cv2.resize(feature_map.numpy(), featmap_ress[0], interpolation=cv2.INTER_CUBIC) # resize (don't normalize!)
            
            feat_maps.append(feature_map)
        
        head_feat_map = np.mean(feat_maps, axis=0) # calc average among all resolutions in one head
        
        head_feat_maps.append(head_feat_map)

    res_feat_map = np.mean(head_feat_maps, axis=0) # calc average among all heads

    # masking
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, featmap_ress[0])
    res_feat_map[np.where(mask == 0)] = 0
    
    return res_feat_map

def cvt_img2tensor(img: np.ndarray) -> torch.Tensor:
    """resize, normalize and convert image to tensor

    Args:
        img (np.ndarray): BGR image

    Returns:
        torch.Tensor: _description_
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pil_img = Image.fromarray(img).convert("RGB")
    image_transformed, _ = transform(pil_img, None)
    return image_transformed

def calc_s1(orgn_pos, length):
    x, y, w, h = orgn_pos
    s1 = length / max(h, w)
    return s1

def calc_delta(orgn_pos, length, s1):
    x, y, w, h = orgn_pos
    res_w, res_h = w * s1, h * s1
    delta = [(length - res_w) / 2 - s1*x, (length - res_h) / 2 - s1*y]
    return delta

def calc_A_F2I(s1, attention_map_s, delta) -> np.ndarray:
    A_F2I = np.array([[1/(s1*attention_map_s), 0, -delta[0]/s1], 
                      [0, 1/(s1*attention_map_s), -delta[1]/s1],
                      [0, 0, 1]])
    return A_F2I

class AttentionMapMetaData:
    def __init__(self, orgn_pos: List[int], length: int,  id: int=None, attention_map_length: int=100):
        """
        Args:
            orgn_pos ([x: int, y: int, w: int, h: int]): position of object in original image
            length (int): length of clopped square image
        
        Variables:
            attention_map (np.ndarray): attention map whose size is (100, 100) as default.
            s1 (float): scale of resizing from input_image_scale to (length, length), described as length / max{h, w}
            delta ([delta_x: float, delta_y: float]): parallel movement
            attention_map_s (float): scale of resizing from (length, length) to (100, 100), described as 100 / length
            cluster_label (int): cluster label of attention map
        """
        self.id = id
        self.s1 = calc_s1(orgn_pos, length) # scale of resizing from input_image_scale to (length, length), described as length / max{w, h}
        self.delta = calc_delta(orgn_pos, length, self.s1) # parallel movement on a resized and clopped square image
        self.attention_map_s = attention_map_length / length # scale of resizing from (length, length) to (100, 100), described as 100 / length
        self.A_F2I = calc_A_F2I(self.s1, self.attention_map_s, self.delta) # 3x3 matrix for converting feature map coordinates to input image coordinates
        self.leaf_root_crd = None # coordinate of root of attention map in input image coordinate system
        self.leaf_tip_crd = None # coordinate of tip of attention map in input image coordinate system
        self.cluster_label = None # a label of a cluster this attention map belongs to
        self.gt_plant_center = None # ground truth plant center in input image coordinate system
        self.ref_point = None # reference point to determine leaf root in input image coordinate system
    
    def set_s1(self, orgn_pos, length):
        x, y, w, h = orgn_pos
        self.s1 = length / max(h, w)
        return self.s1
    def set_delta(self, orgn_pos, length):
        x, y, w, h = orgn_pos
        if self.s1 is None:
            self.s1 = length / max(h, w)
        res_w, res_h = int(w * self.s1), int(h * self.s1)
        self.delta = [int((length - res_w) / 2 - self.s1*x), int((length - res_h) / 2 - self.s1*y)]
        return self.delta
    def cvt2orgn_crd(self, feature_map_crd):
        """convert feature map coordinate to original image coordinate

        Args:
            attention_map_crd (_type_): _description_
        """
        homo_attention_map_crd = np.array([[feature_map_crd[0]], [feature_map_crd[1]], [1]])
        homo_input_image_crd = self.A_F2I @ homo_attention_map_crd
        u, v = homo_input_image_crd[0][0], homo_input_image_crd[1][0]
        return round(u), round(v)
    
def saveAttentionMapMetadata_list(save_path: str, leaf_masks_data: List[AttentionMapMetaData]) -> None:
    """save AttentionMapMetaData list as csv file

    Args:
        save_path (str): _description_
        leaf_masks_data (List[AttentionMapMetaData]): _description_
    """
    
    save_data = []
    for leaf_mask_data in leaf_masks_data:
        id = leaf_mask_data.id
        root_crd = leaf_mask_data.leaf_root_crd
        tip_crd = leaf_mask_data.leaf_tip_crd
        ref_crd = leaf_mask_data.ref_point
        gt_plant_center = leaf_mask_data.gt_plant_center
        row = {'id': id, 'root_crd': root_crd, 'tip_crd': tip_crd, 'ref_crd': ref_crd, 'gt_plant_center': gt_plant_center}
        save_data.append(row)
    save_dict_list_as_csv(save_path, save_data)
    
    return