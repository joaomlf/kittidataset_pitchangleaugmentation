from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.kitti_yolo_dataset import KittiYOLODataset
from utils.pytorchtools import EarlyStopping

from terminaltables import AsciiTable
import os, sys, time, datetime, argparse

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/complex_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--model_def", type=str, default="config/complex_yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--pretrained_weights", type=str, default="checkpoints/yolov3-tiny_benchmark.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok = True)
    class_names = load_classes("data/classes.names")

    # Initiate model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get training dataset
    train_set = KittiYOLODataset(
        cnf.root_dir,
        split = 'train',
        mode = 'TRAIN',
        folder = 'training',
        data_aug = True,
        multiscale = opt.multiscale_training )

    train_loader = DataLoader(
        train_set,
        opt.batch_size,
        shuffle = True,
        num_workers = opt.n_cpu,
        pin_memory = True,
        collate_fn = train_set.collate_fn )

    # Get validation dataset
    valid_set = KittiYOLODataset(
        cnf.root_dir,
        split = 'valid',
        mode = 'TRAIN',
        folder = 'training',
        data_aug = False )

    valid_loader = DataLoader(
        valid_set,
        opt.batch_size,
        shuffle = True,
        num_workers = 1,
        collate_fn = valid_set.collate_fn )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "im",
        "re",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj" ]

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=10, verbose=True, path='checkpoints/yolov3_normal-exp2-all_classes.pth')
    early_stopping = EarlyStopping(patience=10, verbose=True, path='checkpoints/yolov3_tiny-exp1-cyclist.pth')

    start_time_total = time.time()

    for epoch in range(1, opt.epochs+1, 1):
        start_time = time.time()

        ###################
        # train the model #
        ###################
        model.train()
        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            batches_done = len(train_loader) * epoch + batch_i
            # optimizer.zero_grad()

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            if targets.size()[0] == 0: # no real targets/objects
                print("!!! Ignoring current batch due to no real targets/objects !!!")
                continue # ignore current batch

            loss, outputs = model(imgs, targets)

            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            # optimizer.step()

            train_losses.append(loss.item())

            # Log progress
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i+1, len(train_loader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_loader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)

            elapse_time = round( time.time() - start_time_total, 2)
            print( "Elapsed time = ", time.strftime( "%H:%M:%S", time.gmtime(elapse_time) ) )

            model.seen += imgs.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        print("Validating model...")
        for batch_i, (_, imgs, targets) in enumerate(valid_loader):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(opt.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{opt.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f}, ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("!!! EARLY STOPPING !!!")
            break

    elapse_time = round( time.time() - start_time_total, 2)
    print( "Total time = ", time.strftime( "%H:%M:%S", time.gmtime(elapse_time) ) )

    # visualize the loss as the network trained
    plt.figure(figsize=(10,8))
    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    lowest_valid = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(lowest_valid, linestyle='--', color='r', label='Early Stopping Checkpoint')

    # plt.title('Complex-YOLOv3 NORMAL - Dataset ALL CLASSES')
    plt.title('Complex-YOLOv3 TINY - Dataset CYCLIST')

    plt.xlabel('Epoch #')
    plt.ylabel('Loss')

    #plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # plt.savefig('loss_plot-normal-exp2-all_classes.png')
    plt.savefig('loss_plot-tiny-exp1-cyclist.png')

    # plt.show()
