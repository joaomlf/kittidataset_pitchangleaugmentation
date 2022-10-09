from __future__ import division

from models import *
from utils.eval_utils import *

import os, sys, time, datetime, argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import utils.config as cnf
from utils.kitti_yolo_dataset import KittiYOLODataset

def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    target_classes = []
    sample_metrics = []

    min_fps = 50
    max_fps = 0
    avg_fps = 0

    model.eval()
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        target_classes += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            start_time = time.time()
            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            end_time = time.time()

        total_time = end_time - start_time

        a = 1.0 / total_time
        avg_fps = avg_fps + a
        if a < min_fps:
            min_fps = a
        elif a > max_fps:
            max_fps = a

        sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels, similarities = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    ap, aos, unique_classes = metrics_per_class(true_positives, pred_scores, pred_labels, similarities, target_classes)

    return ap, aos, unique_classes, min_fps, max_fps, avg_fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    # NORMAL
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_normal-exp2-all_classes.pth", help="path to weights file")
    #
    # TINY
    # parser.add_argument("--model_def", type=str, default="config/complex_yolov3-tiny.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_tiny-exp1-all_classes.pth", help="path to weights file")
    #
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data_config = parse_data_config(opt.data_config)
    #class_names = load_classes(data_config["names"])
    class_names = load_classes(opt.class_path)

    # Initiate model
    model = Darknet(opt.model_def).to(device)

    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

    # Get evaluation dataset
    dataset = KittiYOLODataset(
        cnf.root_dir,
        split = 'eval',
        mode = 'EVAL',
        folder = 'training',
        data_aug = False )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        opt.batch_size,
        shuffle = True,
        num_workers = 1,
        collate_fn = dataset.collate_fn )

    ######################
    # evaluate the model #
    ######################
    AP, AOS, ap_class, min_fps, max_fps, avg_fps = evaluate(
        model,
        dataloader,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size )

    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]})\nAP: {AP[i]}\nAOS: {AOS[i]}\n")

    print("Min. FPS: "+"{:.2f}".format(min_fps))
    print("Max. FPS: "+"{:.2f}".format(max_fps))
    print("Avg. FPS: "+"{:.2f}".format(avg_fps/1496))
