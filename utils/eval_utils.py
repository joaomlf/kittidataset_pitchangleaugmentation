import math
import tqdm
import torch
import numpy as np
import utils.kitti_bev_utils as bev_utils

from shapely.geometry import Polygon

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")

    names = fp.read().split("\n")[:-1]

    return names

def non_max_suppression_rotated_bbox(prediction, conf_thres=0.95, nms_thres=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 6] * image_pred[:, 7:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            #large_overlap = rotated_bbox_iou(detections[0, :6].unsqueeze(0), detections[:, :6], 1.0, False) > nms_thres # not working
            large_overlap = rotated_bbox_iou_polygon(detections[0, :6], detections[:, :6]) > nms_thres
            large_overlap = torch.from_numpy(large_overlap.astype('uint8'))
            # large_overlap = torch.from_numpy(large_overlap)
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 6:7]
            # Merge overlapping bboxes by order of confidence
            detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """
        Compute true positives, predicted scores and predicted labels per sample
    """

    # print("Total number of outputs = ", len(outputs))
    # print("Total number of targets = ", targets.shape[0])

    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])
        similarities = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        # print("Local number of targets = ", annotations.shape[0])

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                # print("TP index :", box_index.detach().cpu().numpy())
                # print("GT detected annotation: ", target_boxes[box_index,:].detach().cpu().numpy())

                # deltas = calculate_deltas(pred_box, target_boxes)
                #
                # tmp = np.zeros(len(deltas))
                # for i in range(len(tmp)):
                #     tmp[i] = (1.0 + np.cos(deltas[i])) / 2.0

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

                    # only calculate AOS for TPs; AOS for FPs = 0
                    delta = calculate_delta(pred_box, target_boxes[box_index])
                    similarities[pred_i] = (1.0 + np.cos(delta)) / 2.0

                    # print("Pred_i :", pred_i)
                    # print("Delta :", delta)

        # print("Similarities :", similarities)

        batch_metrics.append([true_positives, pred_scores, pred_labels, similarities])

    return batch_metrics


def rotated_bbox_iou_polygon(box1, box2):
    box1 = box1.detach().cpu().numpy()
    box2 = box2.detach().cpu().numpy()

    x, y, w, l, im_yaw, re_yaw = box1
    alpha = np.arctan2(im_yaw, re_yaw)
    bbox1 = np.array(bev_utils.get_corners(x, y, w, l, alpha)).reshape(-1,4,2)
    bbox1 = convert_format(bbox1)

    bbox2 = []
    for i in range(box2.shape[0]):
        x, y, w, l, im_yaw, re_yaw = box2[i,:]
        alpha = np.arctan2(im_yaw, re_yaw)
        bev_corners = bev_utils.get_corners(x, y, w, l, alpha)
        bbox2.append(bev_corners)

    bbox2 = convert_format(np.array(bbox2))

    return compute_iou(bbox1[0], bbox2)


def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]

    return np.array(polygons)


def compute_iou(box, boxes):
    """
    Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """

    # Calculate intersection areas
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]

    return np.array(iou, dtype=np.float32)


def calculate_delta(pred_box, target_box):
    _, _, _, _, pred_im, pred_re = pred_box.detach().cpu().numpy()
    _, _, _, _, target_im, target_re = target_box.detach().cpu().numpy()

    delta = np.arctan2(target_im, target_re) - np.arctan2(pred_im, pred_re)

    return delta


def metrics_per_class(tp, conf, pred_class, similarities, target_class):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_class: Predicted object classes (list).
        target_class: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)

    tp, conf, pred_class, similarities = tp[i], conf[i], pred_class[i], similarities[i]

    # Find unique classes
    unique_classes = np.unique(target_class)

    # Create Precision-Recall curve and compute AP and AOS for each class
    ap, aos, p, r = [], [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP and AOS..."):
        i = pred_class == c
        n_gt = (target_class == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            aos.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs, TPs and Similarities
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
            simc = (similarities[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Similarity
            similarity_curve = simc / (tpc + fpc)

            # AOS from recall-similarity curve
            aos.append(compute_aos(recall_curve, similarity_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    # p, r, ap = np.array(p), np.array(r), np.array(ap)
    # f1 = 2 * p * r / (p + r + 1e-16)

    return ap, aos, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # ap = (ap / 11) * 100
    ap = ap * 100

    return ap


def compute_aos(recall, similarity):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    msim = np.concatenate(([0.0], similarity, [0.0]))

    # compute the similarity envelope
    for i in range(msim.size - 1, 0, -1):
        msim[i - 1] = np.maximum(msim[i - 1], msim[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    aos = np.sum((mrec[i + 1] - mrec[i]) * msim[i + 1])
    # aos = (aos / 11) * 100
    aos = aos * 100

    return aos
