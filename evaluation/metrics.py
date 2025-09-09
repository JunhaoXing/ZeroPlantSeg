import numpy as np
import torch

def area(mask: np.ndarray):
    return np.count_nonzero(mask)

def overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    return intersection_area

def maskIoU(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_pq_single_class(prediction: np.ndarray, groundtruth: np.ndarray):
    numerator_iou = np.array(0.)
    class_matches = 0
    unmatched_class_predictions = 0

    # take the non-zero INSTANCE labels for both prediction and groundtruth
    labels = np.unique(prediction)
    gt_labels = np.unique(groundtruth)
    labels = labels[labels > 0]
    gt_labels = gt_labels[gt_labels > 0]

    gt_masks = [groundtruth == gt_label for gt_label in gt_labels]
    gt_areas = [(groundtruth == gt_label).sum() for gt_label in gt_labels]

    not_matched = [True] * len(gt_labels)

    for label in labels: # for each predicted label
        iou = np.array(0.)
        best_idx = 0

        pred_mask = (prediction == label)
        pred_area = (prediction == label).sum()

        for idx in np.where(not_matched)[0]: # for each gt label still not matched
            gt_mask = gt_masks[idx]
            gt_area = gt_areas[idx]
            # compute iou with all instance gt labels and store the best
            intersection = ((pred_mask & gt_mask).sum()).astype(np.float64)
            union = (pred_area + gt_area - intersection).astype(np.float64)

            iou_tmp = intersection / union
            if iou_tmp > iou:
                iou = iou_tmp
                best_idx = idx

        # if the best iou is above 0.5, store the match pred_label-gt_label-iou
        if iou > 0.5:
            class_matches += 1
            numerator_iou += iou
            not_matched[best_idx] = False
        else:
            # unmatched_class_predictions.append(label.item())
            unmatched_class_predictions += 1

    true_positives = class_matches # len(class_matches)     # TP = number of matches
    # FP: number of unmatched predictions
    false_positives = unmatched_class_predictions # len(unmatched_class_predictions)
    # FN: number of unmatched gt labels
    false_negatives = len(gt_labels) - class_matches # len(class_matches)
    
    if true_positives + false_positives + false_negatives != 0:
        panoptic_quality_one_class = numerator_iou / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)
    else:
        panoptic_quality_one_class = None
    
    if true_positives != 0:
        segmentation_quality_one_class = numerator_iou / true_positives
    else:
        segmentation_quality_one_class = None
    
    if true_positives + false_positives + false_negatives != 0:
        recognition_quality_one_class = true_positives / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)
    else:
        recognition_quality_one_class = None

    return panoptic_quality_one_class, segmentation_quality_one_class, recognition_quality_one_class

def PQ(gt_ps_img: np.ndarray, pred_ps_img: np.ndarray, gt_visibility: torch.Tensor=None):
    """
    Calculate Panoptic Quality, Segmentation Quality, Recognition Quality. Note that all the metrics are in [0, 1], not in [0, 100] [%].
    
    Args: 
        gt_ps_img (np.ndarray): gt panoptic segmentation img
        pred_ps_img (np.ndarray): pred panoptic segmentation img
        gt_visibility (torch.Tensor): visibility of each instance in gt_ps_img. 1 if fully visible, 0 if partially visible.
    
    Returns: 
        PanopticQuality (float)
        SegmentationQuality (float)
        RecognitionQuality (float)
    
    """

    if gt_visibility is not None:
        def filter_partial_masks(pred_instance_masks: torch.Tensor, pred_semantics: torch.Tensor, gt_instance_masks: torch.Tensor, gt_semantics: torch.Tensor, gt_visibility: torch.Tensor):
            assert torch.max(gt_visibility) <= 1.0
            assert torch.min(gt_visibility) >= 0.0

            gt_instance_partial_masks = []
            # set semantics of partial masks to 0 and store partial_masks
            for gt_instance_id in torch.unique(gt_instance_masks):
                if gt_instance_id == 0:
                    continue
                gt_mask = gt_instance_masks == gt_instance_id
                gt_vis = torch.unique(gt_visibility[gt_mask]) # contain a single value
                if gt_vis > 0.5:
                    continue
                gt_semantics[gt_mask] = 0
                gt_instance_partial_masks.append(gt_mask)


            for pred_instance_id in torch.unique(pred_instance_masks):
                if pred_instance_id == 0:
                    continue
                pred_mask = pred_instance_masks == pred_instance_id

                for gt_mask in gt_instance_partial_masks:
                    # compute how much of the prediction is within the ground truth
                    a = torch.sum(gt_mask & pred_mask)
                    b = torch.sum(pred_mask)
                    score = a / (b + 1e-12)
                    assert score <= 1.0

                    if score > 0.5:
                        pred_semantics[pred_mask] = 0
            return
        gt_ps_img_tensor = torch.tensor(gt_ps_img.astype(np.int16))
        pred_ps_img_tensor = torch.tensor(pred_ps_img.astype(np.int16))
        
        pred_semantic = torch.zeros_like(pred_ps_img_tensor)
        pred_semantic[pred_ps_img != 0] = 1
        gt_semantic = torch.zeros_like(gt_ps_img_tensor)
        gt_semantic[gt_ps_img != 0] = 1
        
        filter_partial_masks(pred_ps_img_tensor, pred_semantic, gt_ps_img_tensor, gt_semantic, gt_visibility)
        
        pred_ps_img_tensor = (pred_ps_img_tensor * pred_semantic).int()
        gt_ps_img_tensor = (gt_ps_img_tensor * gt_semantic).int()
    
    #PanopticQuality, SegmentationQuality, RecognitionQuality = compute_pq_single_class(pred_ps_img_tensor, gt_ps_img_tensor)
    PanopticQuality, SegmentationQuality, RecognitionQuality = compute_pq_single_class(pred_ps_img, gt_ps_img)
    
    return PanopticQuality, SegmentationQuality, RecognitionQuality