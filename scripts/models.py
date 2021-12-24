#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Comparision experiments
: standalone segmentation using torch in-built models - fcn, deeplabv3, lr-aspp
: Author - Xi Mo
: Institute - University of Kansas
: Date - 5/13/2021 last updated 5/15/2021
: Model Reference:
    https://pytorch.org/vision/stable/models.html#semantic-segmentation
: HowTo:
    0) This script adopts functoins from RGANet codes, can be execuated independently.
    Requirments: pytorch >= 1.0.0, python >= 3.6, numpy
    1) To specify parameters, refer to CONFIG for details.
    2) You can copy and rename this script to run several different models at a time,
    to do this, you must specify correct gpu using '-gpu' parameter (default: 0),
    3) You may change optimizer and parameters in triaing part.
    4) Don't forget to move 'evaluation.txt' before running another evaluation.
"""

import torch
import time
import argparse
import math
import numpy as np
import torch.nn.functional as ops

from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset
from PIL import Image

parser = argparse.ArgumentParser(description="Arguments for training, validation and testing")

parser.add_argument("-gpu", type = int, default = 0,
                    help = "Designate GPU # for trainig and testing")
# Training
parser.add_argument("-train", action = "store_true",
                    help = "Train network only")
parser.add_argument("-r", "--restore", action = "store_true",
                    help = "Restore training by loading specified checkpoint or lattest checkpoint")
# Accomodation to suction dataset
parser.add_argument("-i", "--image", type = Path,
                    default = r"../dataset/suction-based-grasping-dataset/data/color-input",
                    help = "Directory to training images")
parser.add_argument("-l", "--label", type = Path,
                    default = r"../dataset/suction-based-grasping-dataset/data/label",
                    help = "Directory to training annotations")
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by users")
parser.add_argument("-d", "--dir", type = Path, default = r"results",
                    help = r"valid for test mode, specify the folder to save results and labels, or "
                           r"valid for validate mode, specify the source to load images/save results")

# Testing
parser.add_argument("-test", action = "store_true",
                    help = "Test and visualize only")
parser.add_argument("-v", "--validate", action = "store_true",
                    help = "Validate tesing results using metrics")

CONFIG = {
    "MODEL": "deeplab",            # choose between "fcn", "deeplab", "lr-aspp"
    "BACKBONE": "mobilenet",        # backbone "resnet50", "resnet101" for fcn
                                   # backbone "resnet50", "resnet101", "mobilenet" for "deeplab"
                                   # backbone "mobilenet" for "lr-aspp" (this will ignore backbone
                                   # setting)

    # Training
    "DOUBLE": False,            # double the size of training set, turn off when "AUGMENT" is False
    "AUGMENT": False,           # switch to enable augmentation, valid regardless of "DOUBLE"
    "AUX_LOSS": False,          # whether to apply auxiliary loss during training
    "PRETRAIN": False,          # use pre-trained weights from COCO train2017 (21 classes as Pascal VOC)
    "SHOW_PROG": True,          # displays a progress bar of the download if True
    "BATCHSIZE": 18,             # batchsize for training
    "EPOCHS": 300,               # epoches for training
    "SHOW_LOSS": 10,            # number of minibatchs processed to print training info
    "SAVE_MODEL": 2,            # epoch intervel to save, start counting from epoch 2
    "SHUFFLE": True,            # random shuffle
    "NUM_WORKERS": 0,           # set to 0 if memorry error occurs
    "PIN_MEMORY": True,         # set to false if memory is insufficient
    "DROP_LAST": False,
    "NUM_CLS": 3,               # number of classes
    "INT_CLS": (255, 0, 128),   # raw label intensity levels to differentiate classes

    # Testing
    "DENORM": False,             # set to True to disable testing normalization
    "TEST_BATCH": 20,            # batchsize for testing
    "TEST_MUL": 5,              # set a multiplier for testing
    "TEST_TIME": 1,             # show runtime stats done running certain number of testing batches
    "TEST_WORKERS": 0,          # set number of workers to run testing batches
    "TEST_PIN": True,           # set to True if memory is pinned for testing batches
    "TEST_SAVE": False,          # if tests are done, save results and their labels to disk
    "TEST_RUNTIME": True,       # False to disable runtime test
    "TEST_BGR": True,           # Test: True - target class save as bright color, otherwise cold color
                                # Validation: True - evaluation cls NUM_CLS-1, otherwise cls 0

    # Validation
    "MGD_INTV": (12, 12),       # set intervals (dH, dW) for metric MGRD
    "MGD_BETA": 6,              # set beta for metric MGRD
    "LAB_CLS": 2,               # interested label class to evalate, range from 0 to NUM_CLS-1
    # Augmentation
    "SIZE": (480, 640),         # (H, W)

    "HAS_NORM": True,                          # normailzationm,for samples only
    "PAR_NORM": {"mean": (0.485, 0.456, 0.406), # dictionary format with tuples
                 "std": (0.229, 0.224, 0.225)}, # standalone option

    "HOR_FLIP": True,          # random horizontal flip
    "PH_FLIP": 0.5,            # must be a number in [0, 1]

    "VER_FLIP": True,          # random vertical flip
    "PV_FLIP": 0.5,            # must be a number in [0, 1]

    "SHIFT": True,             # random affine transform, will not affect the label
    "PAR_SFT": (0.2, 0.2),     # must be a tuple if set "SHIFT" to True
    "P_SFT": 0.6,              # probablity to shift

    "ROTATE": True,            # rotate image
    "ROT_DEG": math.pi,        # rotation degree
    "P_ROT": 0.4,              # probability to rotate

    "COLOR_JITTER": True,           # random color random/fixed jitter
    "P_JITTER": 0.2,                # probability to jitter
    "BRIGHTNESS": 0.5,              # random brightness adjustment, float or (float, float)
    "CONTRAST": 0.5,                # random brightness adjustment, float or (float, float)
    "SATURATION": 0.5,              # random saturation adjustment, float or (float, float)
    "HUE": 0.25,                    # random hue adjustment, float or (float, float)

    "BLUR": True,                   # random gaussian blur
    "P_BLUR": 0.3,                  # probability to blur image
    "PAR_BLUR":
        {"kernel": 15,              # kernal size, can be either one int or [int, int]
         "sigma": (0.5, 3.0)}       # sigma, can be single one float
    }

class SuctionDataset(torch.utils.data.Dataset):
    def __init__(self, imgDir, labelDir, splitDir=None, mode="test", applyTrans=False, sameTrans=True):
        super(SuctionDataset).__init__()
        assert len(CONFIG["INT_CLS"]) > 1, "Must be more than 1 class"
        assert len(CONFIG["INT_CLS"]) == CONFIG["NUM_CLS"], "Number of class does not match intensity levels"
        assert len(CONFIG["SIZE"]) == 2, "Invalid SIZE format"
        assert type(CONFIG["PAR_SFT"]) == tuple and len(CONFIG["PAR_SFT"]) == 2, "Invalid SHIFT parameters"
        assert type(CONFIG["PAR_NORM"]) == dict, "Mean and std must be presented in a dict"
        self.applyTran = applyTrans
        self.sameTrans = sameTrans
        self.mode = mode
        # prepare for FCN training set
        if mode in ["train", "test"]:
            if splitDir and labelDir:
                self.img = self.read_split_images(imgDir, splitDir, ".png", 1)
                self.imgLen = len(self.img)
                assert self.imgLen, "Empty dataset, please check directory"
                self.nameList = list(self.img.keys())
                self.W, self.H = self.img[self.nameList[0]].size
                self.label = self.read_split_images(labelDir, splitDir, ".png", 0)
            else:
                raise IOError("Must specify training split file and annotation directory")
        # prepare for validation. NOTE: network ONLY supports color samples and greyscale labels
        if mode == "validate":
            self.img = self.read_image_from_disk(imgDir, colorMode = 1)
            self.imgLen = len(self.img)
            assert self.imgLen, "Empty dataset, please check directory"
            self.nameList = list(self.img.keys())
            self.W, self.H = self.img[self.nameList[0]].size
            self.label = self.read_image_from_disk(labelDir, colorMode = 0)

    # get one pair of samples
    def __getitem__(self, idx):
        imgName = self.nameList[idx]
        img, label = self.img[imgName], self.label[imgName]
        # necesary transformation
        operate = transforms.Compose([transforms.ToTensor(), self._transform_pad_image()])
        img = operate(img)
        label = self._convert_img_to_uint8_tensor(label)
        # optical transformation
        img = self.img_normalize(img)
        img = self.img_random_color_jitter(img)
        img = self.img_random_blur(img)
        img, label = self.img_random_flip(img, label)
        img, label = self.img_random_shift_rotate(img, label)
        return img, label

    # get length of total smaples
    def __len__(self):
        return self.imgLen

    # read names/directories from text files
    @classmethod
    def read_image_id(cls, filePath: Path, postFix: str) -> [str]:
        assert filePath.is_file(), f"Invalid file path:\n{filePath.absolute()}"
        with open(filePath, 'r') as f:
            imgNames = f.readlines()
        return [] if not imgNames else [ _.strip()+postFix for _ in imgNames]

    # directly read image from directory
    @classmethod
    def read_image_from_disk(cls, folderPath: Path, colorMode=1) -> {str: Image.Image}:
        imgList = folderPath.glob("*")
        return cls.read_image_data(imgList, colorMode)

    # read a bunch of images from a list of image paths
    @classmethod
    def read_image_data(cls, imgList: [Path], colorMode=1) -> {str: Image.Image}:
        dump = {}
        for imgPath in imgList:
            assert imgPath.is_file(), f"Invalid image path: \n{imgPath.absolute()}"
            img = Image.open(imgPath)
            if not colorMode: img = img.convert('L')
            dump[imgPath.stem] = img
        return dump

    # read images according to split lists
    @classmethod
    def read_split_images(cls, imgRootDir: Path, filePath: Path, postFix=".png", colorMode=1) -> {str: Path}:
        imgList = cls.read_image_id(filePath, postFix)
        imgList = [imgRootDir.joinpath(_) for _ in imgList]
        return cls.read_image_data(imgList, colorMode)

    # PIL label to resized tensor
    def _convert_img_to_uint8_tensor(self, label: Image) -> torch.Tensor:
        dummy = np.array(label, dtype = np.uint8)
        assert dummy.ndim == 2, "Only for grayscale labelling images"
        save = []
        if self.mode in ["trainGCRF", "testGCRF"]:
            intLevels = CONFIG["INTCL_GCRF"]
        else:
            intLevels = CONFIG["INT_CLS"]

        for idx, val in enumerate(intLevels):
            save.append(np.where(dummy == val))
        for idx, val in enumerate(save):
            dummy[val] = idx
        dummy = torch.tensor(dummy, dtype = torch.uint8)
        dummy = self._transform_pad_image()(dummy)
        return dummy

    # one-hot encoder for int64 label tensor
    @staticmethod
    def one_hot_encoder(label: torch.Tensor) -> torch.Tensor:
        assert len(label.shape) == 3, r"Length of the tensor must be [batchSize, H, W]"
        label = label.to(torch.int64)
        dummy = torch.nn.functional.one_hot(label, CONFIG["NUM_CLS"])
        return dummy.permute(0, 3, 1, 2).to(torch.float32)

    # write testing results to disk
    @staticmethod
    def save_results(batch: torch.Tensor, folder: Path, name: int, postfix=".png", bgr=False):
        assert len(batch.shape) in [3, 4], r"Must be 4-dim/3-dim tensor for color/greyscale images"
        data = batch.cpu().clone()
        for idx in range(data.shape[0]):
            file = str(name + idx) + postfix
            filepath = folder.joinpath(file)
            if len(data.shape) == 4:
                img = data[idx, :, :, :]
                img = img / (img.max() + 1e-31)
                if bgr:
                    index = [2, 1, 0]
                    img = img[index]
            elif len(data.shape) == 3:
                img = data[idx, :, :]
                img = img.to(torch.float32)
                img = img / (CONFIG["NUM_CLS"] - 1 + 1e-31)

            save_image(img, filepath)
        return

    # padding
    def _transform_pad_image(self):
        H, W =  CONFIG["SIZE"]
        dH, dW = max(0, H-self.H), max(0, W-self.W)
        padding = [dW//2, dH//2, dW-dW//2, dH-dH//2]
        return transforms.Pad(padding=padding, padding_mode='constant')

    # shift and rotation
    def img_random_shift_rotate(self, img: torch.Tensor, label: torch.Tensor)-> [torch.Tensor, torch.Tensor]:
        if CONFIG["SHIFT"] or CONFIG["ROTATE"]:
            if self.applyTran:
                if random.random() < CONFIG["P_SFT"]:
                    DEG = CONFIG["ROT_DEG"] if CONFIG["ROTATE"] else 0
                    SHIFT = CONFIG["PAR_SFT"] if CONFIG["SHIFT"] else None
                    state = torch.get_rng_state()
                    operator = transforms.RandomAffine(DEG, SHIFT)
                    img = operator(img)
                    if self.sameTrans:
                        label = label.unsqueeze(0)
                        torch.set_rng_state(state)
                        label = operator(label)
        return img, label.squeeze(0)

    # random horizontal and vertical flip
    def img_random_flip(self, img: torch.Tensor, label: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        if self.applyTran:
            if CONFIG["HOR_FLIP"]:
                if random.random() < CONFIG["PH_FLIP"]:
                    img = transforms.functional.hflip(img)
                    if self.sameTrans: label = transforms.functional.hflip(label)
            if CONFIG["VER_FLIP"]:
                if random.random() < CONFIG["PV_FLIP"]:
                    img = transforms.functional.vflip(img)
                    if self.sameTrans: label = transforms.functional.vflip(label)
        return img, label

    # color-jitter
    def img_random_color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        if self.applyTran:
            if random.random() < CONFIG["P_JITTER"]:
                operator = transforms.ColorJitter(CONFIG["BRIGHTNESS"], CONFIG["CONTRAST"], CONFIG["HUE"])
                img = operator(img)
        return img

    # normalization
    @classmethod
    def img_normalize(cls, img: torch.Tensor):
        if CONFIG["HAS_NORM"]:
            operator = transforms.Normalize(CONFIG["PAR_NORM"]["mean"], CONFIG["PAR_NORM"]["std"])
            img = operator(img)
        return img

    # Gaussian blur
    def img_random_blur(self, img: torch.Tensor):
        if self.applyTran:
            if CONFIG["BLUR"]:
                if random.random() < CONFIG["P_BLUR"]:
                    operator = transforms.GaussianBlur(CONFIG["PAR_BLUR"]["kernel"],
                                                       CONFIG["PAR_BLUR"]["sigma"])
                    img = operator(img)
        return img


# Helper for training fcn
def train_model_img_to_label(_net, _input, _gtLabel, _optimizer, _lossFuncLabel):
    start_time = time.time()
    _optimizer.zero_grad()
    output = _net(_input)["out"]
    lossLabel = _lossFuncLabel(output, _gtLabel)
    lossLabel.backward()
    _optimizer.step()
    end_time = time.time()
    runtime = (end_time - start_time) * 1e3
    return output, lossLabel.item(), runtime


# Helper to read images, valid for .png images
def read_image_from_disk(path: Path, isTensor=True, colorImg=True) -> {str: torch.Tensor}:
    img = {}
    for imgFile in sorted(path.rglob("*.png")):
        if colorImg:
            imgData = Image.open(imgFile)
        else:
            imgData = Image.open(imgFile).convert('L')

        if isTensor:
            img[imgFile.name] = transforms.ToTensor()(imgData)
        else:
            img[imgFile.name] = np.array(imgData)
    return img


# Helper to in-place transforming grayscale labels to classes
def trans_img_to_cls(img: {str:np.ndarray}) -> {str:np.ndarray}:
    for key, im in img.items():
        save = []
        for idx, val in enumerate(sorted(CONFIG["INT_CLS"])):
            save.append(np.where(im == val))
        for idx, val in enumerate(save):
            im[val] = idx
        img[key] = im
    return img


# Helper to save images
def save_image_to_disk(img:torch.Tensor, path: Path):
    data = img.cpu().clone()
    if len(data.shape) == 3:
        data = data.to(torch.float32)
        data = data / (CONFIG["NCLS_GCRF"] - 1 + 1e-31)
        save_image(data, path)


# Helper to select model
def model_paser():
    if CONFIG["MODEL"] == "fcn":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(pretrained=CONFIG["PRETRAIN"], progress=CONFIG["SHOW_PROG"],
                                 aux_loss=CONFIG["AUX_LOSS"], num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import fcn_resnet101
            model = fcn_resnet101(pretrained=CONFIG["PRETRAIN"], progress=CONFIG["SHOW_PROG"],
                                  aux_loss=CONFIG["AUX_LOSS"], num_classes=CONFIG["NUM_CLS"])
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for FCN.")
    elif CONFIG["MODEL"] == "deeplab":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=CONFIG["PRETRAIN"], progress=CONFIG["SHOW_PROG"],
                                       aux_loss=CONFIG["AUX_LOSS"], num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import deeplabv3_resnet101
            model = deeplabv3_resnet101(pretrained=CONFIG["PRETRAIN"], progress=CONFIG["SHOW_PROG"],
                                        aux_loss=CONFIG["AUX_LOSS"], num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "mobilenet":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
            model = deeplabv3_mobilenet_v3_large(pretrained=CONFIG["PRETRAIN"],
                                                 progress=CONFIG["SHOW_PROG"],
                                                 aux_loss=CONFIG["AUX_LOSS"],
                                                 num_classes=CONFIG["NUM_CLS"])
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for DeepLabv3.")
    elif CONFIG["MODEL"] == "lr-aspp":
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        CONFIG["BACKBONE"] = "mobilenet"
        model = lraspp_mobilenet_v3_large(pretrained=CONFIG["PRETRAIN"], progress=CONFIG["SHOW_PROG"],
                                          aux_loss=CONFIG["AUX_LOSS"], num_classes=CONFIG["NUM_CLS"])
    else:
        raise NameError(f"Unsupported network \"{CONFIG['MODEL']}\" for now, I'd much appreciate "
                        f"if you customize more state-of-the-arts architectures.")
    return model

# Write model to disk
def save_model(baseDir: Path, network: torch.nn.Module, epoch: int,
               optimizer: torch.optim, postfix=CONFIG['MODEL']):
    date = time.strftime(f"%Y%m%d-%H%M%S-Epoch-{epoch}_{postfix}_{CONFIG['BACKBONE']}.pt",
                         time.localtime())
    path = baseDir.joinpath(date)
    print("\nNow saveing model to:\n%s" %path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    print("Done!\n")

# focal loss
class focalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, weights=None, reduction="mean"):
        assert reduction in ["none", "mean", "sum"], "Invalid reduction option"
        super(focalLoss, self).__init__()
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.bufferFlag = False
        self.reduction = reduction
        if weights is not None:
            self.register_buffer("weights", torch.cuda.FloatTensor(weights))
            self.CELoss = torch.nn.CrossEntropyLoss(weight=self.weights, reduction="none")
            self.bufferFlag = True
        else:
            self.CELoss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target):
        ce_loss = self.CELoss(output, target)
        if self.bufferFlag:
            prob = torch.exp(-ops.cross_entropy(output, target, reduction="none"))
        else:
            prob = torch.exp(ce_loss)
        # for with weights or weights is None
        focal_loss = (1 - prob) ** self.gamma * ce_loss
        if self.bufferFlag: focal_loss /= self.weights.sum()
        # default: return mean
        if self.reduction == "mean": return focal_loss.mean()
        if self.reduction == "sum": return  focal_loss.sum()

        return focal_loss

class Metrics:
    def __init__(self, label, gt, cls=CONFIG["NUM_CLS"]-1 , one_hot=False):
        assert gt.ndim == 2, "groundtruth must be grayscale image"
        # one_hot label to unary
        if one_hot:
            assert label.ndim == 3, "label must be 3-dimensional for one-hot encoding"
            label = np.argmax(label, axis = 2)
        else:
            assert label.ndim == 2, "label must be 2-dimensional"

        self.H, self.W = CONFIG["SIZE"]
        label_area, gt_area = np.where(label == cls), np.where(gt == CONFIG["LAB_CLS"])
        self.label_area = set(zip(label_area[0], label_area[1]))
        self.gt_area = set(zip(gt_area[0], gt_area[1]))
        self.TP_FN = len(self.gt_area)
        self.TP_FP = len(self.label_area)
        self.TP = len(self.label_area.intersection(self.gt_area))
        self.FP = self.TP_FP - self.TP
        self.TN = self.H * self.W - self.TP_FN - self.TP_FP + self.TP
        self.FN = self.TP_FN - self.TP

    def IOU(self) -> np.float32:
        UN = self.TP_FN + self.TP_FP
        if UN == 0: return 1.0
        return np.float32(self.TP / (UN - self.TP + 1e-31))

    def ACC(self) -> np.float32:
        accuracy = (self.TP + self.TN) / (self.H * self.W)
        return np.float32(accuracy)

    def DICE(self) -> [np.float32]:
        if self.TP == self.FN == self.FP == 0: return 1.0
        precision = np.float32(self.TP / (self.TP + self.FP + 1e-31))
        recall = np.float32(self.TP / (self.TP + self.FN + 1e-31))
        dice = np.float32(2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-31))
        return dice

    def MGRID(self, interval=(12, 12), beta=0.5, shift=(0.525, 0.5)) -> np.float32:
        assert beta > 0, "beta must be positive"
        Cm, Fm = shift
        assert 0 < Fm < 1, "Fm must a number betweem 0 and 1"
        T = (Fm / (1 - Fm)) ** 3
        assert T / (1 + T) < Cm < (Fm + T) / (1 + T), "Invalid range of Cm"
        S = (1.0 - Cm) / (1 - Fm) ** 3
        # regulator
        curve = lambda F: S * (F - Fm) ** 3 + Cm
        # grouping
        confMat, dH, dW = {}, interval[0], interval[1]
        # bounded intervals
        if interval[0] < 1:
            dH = 1.0
        elif interval[0] > self.H:
            dH = H

        if interval[1] < 1:
            dW = 1.0
        elif interval[1] > self.W:
            dW = W

        nRow, nCol = np.ceil(self.H / dH), np.ceil(self.W / dW)
        coords = np.mgrid[0: nRow, 0: nCol]
        grids = zip(coords[0].reshape(-1), coords[1].reshape(-1))
        for key in grids:
            confMat[key] = [[], []]  # [label, gt]

        for pos in self.label_area:
            confMat[(pos[0] // dH, pos[1] // dW)][0].append(pos)

        for pos in self.gt_area:
            confMat[(pos[0] // dH, pos[1] // dW)][1].append(pos)
        # calculate running mean
        runMean, cnt = 0, 0
        for key in confMat.keys():
            if confMat[key][0] or confMat[key][1]:
                cnt += 1
                label, gt = set(confMat[key][0]), set(confMat[key][1])
                # calculate F_beta score
                TP = len(label.intersection(gt))
                FP, FN = len(label) - TP, len(gt) - TP
                Fbeta = np.float32(
                    (1 + beta ** 2) * TP / (beta ** 2 * (TP + FN) + TP + FP + 1e-31))
                runMean += curve(Fbeta)

        return np.float32(runMean / cnt) if cnt else 1.0

    # evalute and save results to disk
    '''
        options:
        'all': save all evalutaion metrics to disk
        'iou'/'acc'/'dice'/'mgrid': specify metric to be saved
    '''
    def save_to_disk(self, name: str, path: Path, option="all", interval=(12, 12), beta=0.5):
        path = path.joinpath("evaluation.txt")
        if option == "all":
            with open(path, "a+") as f:
                iou, acc = 100 * self.IOU(), 100 * self.ACC()
                dice, mgrid = 100 * self.DICE(), 100 * self.MGRID(interval, beta)
                f.write(f"{name:} iou:{iou:.2f} acc:{acc:.2f} "
                        f"dice:{dice:.2f} mgrid:{mgrid:.2f}\n")
            return
        # write iou only
        if option == "iou":
            with open(path, "a+") as f:
                iou = 100 * self.IOU()
                f.write(f"{name:s} iou:{iou:.2f}\n")
            return
        # write acc only
        if option == "acc":
            with open(path, "a+") as f:
                acc = 100 * self.self.ACC()
                f.write(f"{name:s} acc:{acc:.2f}\n")
            return
        # write dice only
        if option == "dice":
            with open(path, "a+") as f:
                dice = self.DICE()
                f.write(f"{name:s} dice:{dice:.2f}\n")
            return
        # write mgrid only
        if option == "mgrid":
            with open(path, "a+") as f:
                mgrid = self.MGRID(interval, beta)
                f.write(f"{name:s} mgrid:{mgrid:.2f}\n")
            return


if __name__ == '__main__':
    args = parser.parse_args()
    assert isinstance(args.gpu, int), "invalid numerical format to designate GPU"
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")

    torch.autograd.set_detect_anomaly(True)

    ''' train '''

    if args.train:
        assert 0 < CONFIG["SAVE_MODEL"] <= CONFIG["EPOCHS"], "Invalid interval of screenshot"
        # checkpoint filepath check
        if str(args.checkpoint) != "checkpoint":
            if not args.checkpoint.is_file():
                raise IOError(f"Designated checkpoint file does not exist:\n"
                              f"{args.checkpoint.resolve()}")
            ckptPath = args.checkpoint.resolve()
        # Create checkpoint directory
        ckptDir = Path.cwd().joinpath(CONFIG["MODEL"], "checkpoint")
        ckptDir.mkdir(exist_ok=True, parents=True)
        # get the lattest checkpoint if set to default directory and restore is true
        if str(args.checkpoint) == "checkpoint" and args.restore:
            fileList = sorted(ckptDir.glob("*.pt"), reverse=True, key=lambda item: item.stat().st_ctime)
            if len(fileList) == 0:
                raise IOError(f"Cannot find any checkpoint files in:\n"
                              f"{ckptDir.resolve()}\n")
            else:
                ckptPath = fileList[0]
        # prepare for training dataset
        trainSplitPath = args.image.parent.joinpath("train-split.txt")
        if not trainSplitPath.is_file():
            raise IOError(f"Train-split file does not exist, please download the dataset first:\n"
                          f"{trainSplitPath}")

        if CONFIG["DOUBLE"] and CONFIG["AUGMENT"]:
            trainData_orin = SuctionGrasping(args.image, args.label, trainSplitPath,
                                             mode="train", applyTrans=False)
            trainData_mod = SuctionGrasping(args.image, args.label, trainSplitPath,
                                             mode="train", applyTrans=True, sameTrans=True)
            trainData = data.ConcatDataset([trainData_orin, trainData_mod])
        elif CONFIG["AUGMENT"]:
            trainData = SuctionDataset(args.image, args.label, trainSplitPath,
                                             mode="train", applyTrans=True, sameTrans=True)
        else:
            trainData = SuctionDataset(args.image, args.label, trainSplitPath,
                                             mode="train", applyTrans=False)

        trainSet = torch.utils.data.DataLoader(
                                   dataset      = trainData,
                                   batch_size   = CONFIG["BATCHSIZE"],
                                   shuffle      = CONFIG["SHUFFLE"],
                                   num_workers  = CONFIG["NUM_WORKERS"],
                                   pin_memory   = CONFIG["PIN_MEMORY"],
                                   drop_last    = CONFIG["DROP_LAST"])
        # train
        weight = torch.FloatTensor([.3, .3, .4]).to(device) # cross-entropy
        # weight = None # cross-entropy
        # lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight)  # cross-entropy
        lossFuncLabel = focalLoss(gamma=1.3, weights=weight, reduction="mean")# FocalLoss
        lossFuncLabel = lossFuncLabel.to(device)
        # lossFuncLabel = torch.nn.BCELoss() # BCELoss
        model = model_paser()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00018, weight_decay=0.0,
                                      amsgrad=True)
        model.to(device)
        # load checkpoint if restore is true
        if args.restore:
            checkpoint = torch.load(ckptPath)
            print(f"\nCheckpoint loaded:\n{ckptPath}\n")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lastEpoch = checkpoint['epoch']
        else:
            lastEpoch = 0
        print("==================== Start Training %s ====================\n" % (CONFIG["MODEL"]))
        totalBatch = np.ceil(len(trainData) / CONFIG["BATCHSIZE"])
        for epoch in range(lastEpoch, CONFIG["EPOCHS"]):
            runLossLabel = 0.0
            for idx, data in enumerate(trainSet):
                img = data[0].to(device)
                label = data[1]
                # label = SuctionGrasping.one_hot_encoder(label) # one-hot, for BCEloss, otherwise delete
                label = label.long() # cross-entropy
                label = label.to(device)
                output, lossLabel, runtime = train_model_img_to_label(model, img, label,
                                                                    optimizer, lossFuncLabel)
                runLossLabel += lossLabel
                # print info
                if idx % CONFIG["SHOW_LOSS"] == CONFIG["SHOW_LOSS"] - 1:
                    # Simple evaluation for a batch, class_id: NUM_CLS-1, loss: cross entropy
                    with torch.no_grad():
                        pred = torch.argmax(output.detach(), dim=1)
                        labs = label.detach()
                        TP_FP = len(torch.where(pred == CONFIG["NUM_CLS"] - 1)[0])
                        TP_FN = len(torch.where(labs == CONFIG["NUM_CLS"] - 1)[0])
                        TP = len(torch.where(torch.add(pred, labs) == 2 * (CONFIG["NUM_CLS"] - 1))[0])
                        IU = float(torch.div(TP, TP_FP + TP_FN - TP + 1e-31))
                        precision = float(torch.div(TP, TP_FP + 1e-31))
                        recall = float(torch.div(TP, TP_FN + 1e-31))

                    averLossLabel  = runLossLabel/CONFIG["SHOW_LOSS"]
                    print("Epoch: %2d, batch: %4d/%d, loss: %.5f, runtime: %3f ms/batch, "
                          "Jaccard: %.2f, Precision: %.2f, Recall: %.2f"
                          % (epoch+1, idx+1, totalBatch, averLossLabel, runtime, IU, precision, recall))
                    runLossLabel = 0.0
            # save checkpoint
            if epoch not in [0, CONFIG["EPOCHS"] - 1] and epoch % CONFIG["SAVE_MODEL"] == 0:
                save_model(ckptDir, model, epoch+1, optimizer)
        # save last checkpoint when finished training
        save_model(ckptDir, model, epoch+1, optimizer, CONFIG['MODEL'])
        print("==================== %s Done Training ====================\n" %(CONFIG["MODEL"]))

    ''' Test '''

    if args.test:
        if CONFIG["DENORM"]: CONFIG["HAS_NORM"] = False
        # checkpoint filepath check
        if str(args.checkpoint) != "checkpoint":
            if not args.checkpoint.is_file():
                raise IOError(f"Designated checkpoint file does not exist:\n"
                              f"{args.checkpoint.resolve()}")
            ckptPath = args.checkpoint.resolve()
        else:
            ckptDir = Path.cwd().joinpath(CONFIG["MODEL"], "checkpoint")
            if not ckptDir.is_dir():
                raise IOError(f"Default folder 'checkpoint' does not exist:\n{ckptDir.resolve()}")
            fileList = sorted(ckptDir.glob("*.pt"), reverse=True,key=lambda item: item.stat().st_ctime)
            if len(fileList) == 0:
                raise IOError(f"Cannot find any checkpoint files in:\n{ckptDir.resolve()}\n")
            else:
                ckptPath = fileList[0]

        testSplitPath = args.image.parent.joinpath("test-split.txt")
        if not testSplitPath.is_file():
            raise IOError(f"Test-split file does not exist, please download the dataset first:\n"
                          f"{trainSplitPath}")
        testData = SuctionDataset(args.image, args.label, testSplitPath,
                                   mode="test", applyTrans=False, sameTrans=False)
        testSet = torch.utils.data.DataLoader(
                                  dataset     = testData,
                                  batch_size  = CONFIG["TEST_BATCH"],
                                  shuffle     = False,
                                  num_workers = CONFIG["TEST_WORKERS"],
                                  pin_memory  = CONFIG["TEST_PIN"],
                                  drop_last   = False)
        # test
        model = model_paser()
        checkpoint = torch.load(ckptPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        totalBatch = np.ceil(len(testData) / CONFIG["TEST_BATCH"])
        # get runtime estimation
        if CONFIG["TEST_RUNTIME"]:
            with torch.no_grad():
                if CONFIG["TEST_RUNTIME"]:
                    if CONFIG["TEST_TIME"] < 1: CONFIG["TEST_TIME"] = 1
                    if CONFIG["TEST_MUL"] < 1: CONFIG["TEST_MUL"] = 1
                    tailCount = len(testData) % CONFIG["TEST_BATCH"]
                    totalTime = 0
                    for i in range(CONFIG["TEST_MUL"]):
                        print(f"\nFold {i + 1} of {CONFIG['TEST_MUL']}:\n")
                        for idx, data in enumerate(testSet):
                            img = data[0].to(device)
                            startTime = time.time()
                            _ = model(img)["out"][0]
                            endTime = time.time()
                            batchTime = (endTime - startTime) * 1e3
                            totalTime += batchTime
                            if (idx + 1) % CONFIG["TEST_TIME"] == 0:
                                if idx == len(testSet) - 1 and tailCount:
                                    divider = tailCount
                                else:
                                    divider = CONFIG["TEST_BATCH"]

                                print("batch: %4d/%d, average inference over current batch: %6fms per image"
                                      % (idx + 1, totalBatch, batchTime / divider))

                    print("\n======================== Runtime Test Done ========================\n"
                          "Average (%d images in total): %6fms" % (len(testData) * CONFIG["TEST_MUL"],
                                                                   totalTime / (len(testData) * CONFIG["TEST_MUL"])))

        # save results if required
        if CONFIG["TEST_SAVE"]:
            if str(args.dir) != "results":
                if not args.dir.is_dir():
                    raise IOError(f"Invalid sample folder:\n{args.dir.resolve()}")
            else:
                name = "".join(["results", '_', CONFIG["MODEL"], '_', CONFIG["BACKBONE"]])
                args.dir = Path.cwd().joinpath(CONFIG["MODEL"], name)

            print(f"\nNow saving test results to:\n{args.dir.resolve()}\n")
            labelDir = args.dir.joinpath("annotations")
            resultDir = args.dir.joinpath("output")
            imgDir = args.dir.joinpath("images")
            labelDir.mkdir(exist_ok=True, parents=True)
            resultDir.mkdir(exist_ok=True, parents=True)
            imgDir.mkdir(exist_ok=True, parents=True)
            imgCnt = 1
            for idx, data in enumerate(testSet):
                SuctionDataset.save_results(data[0], imgDir, imgCnt, postfix=".png")
                SuctionDataset.save_results(data[1], labelDir, imgCnt, postfix=".png")
                img = data[0].to(device)
                labelOut = model(img)['out']
                labelOut = torch.softmax(labelOut, dim=1)
                SuctionDataset.save_results(labelOut, resultDir, imgCnt, postfix=".png",
                                            bgr=CONFIG["TEST_BGR"])
                imgCnt += CONFIG["TEST_BATCH"]
                if (idx + 1) % CONFIG["TEST_TIME"] == 0:
                    print(f"%4d/%d batches processed" % (idx + 1, totalBatch))

            print("\n=========== %s Testing Results Saved ============" %(CONFIG["MODEL"]))

    ''' Validate results '''

    if args.validate:
        # check image directory to read results
        if str(args.dir) != "results":
            if not args.dir.is_dir():
                raise IOError(f"Invalid output folder to read from:\n{args.dir.resolve()}")
        else:
            name = "".join(["results", '_', CONFIG["MODEL"], '_', CONFIG["BACKBONE"]])
            args.dir = Path.cwd().joinpath(CONFIG["MODEL"], name)
        # output folder and label folder
        imgDir = args.dir.joinpath("output")
        outDir = args.dir.joinpath("evaluation")
        labDir = args.dir.joinpath("annotations")
        assert labDir.is_dir(), f"Cannot find folder 'annotations' in: \n{labDir.resolve()}"
        print(f"\nLoading outputs and annotations from:\n{args.dir.resolve()}\n")
        outDir.mkdir(exist_ok=True, parents=True)
        imgList = read_image_from_disk(imgDir, isTensor=False)
        # convert label to classes
        labList = read_image_from_disk(labDir, isTensor=False, colorImg=False)
        labList = trans_img_to_cls(labList)
        assert imgList and len(imgList) == len(labList), "Empty folder or length mismatch"
        L, cnt = len(imgList), 0
        for name, img in imgList.items():
            cnt += 1
            start = time.time()
            if CONFIG["TEST_BGR"]:
                metric = Metrics(img, labList[name], cls=0, one_hot=True)  # cls=0: BGR format
            else:
                metric = Metrics(img, labList[name], cls=2, one_hot=True)  # cls=0: BGR format

            metric.save_to_disk(name, outDir)
            end = time.time()
            print(f"{cnt:6d}/{L} evaluated, {(end-start)*1e3:.3f}ms per image")
        print(f"\nEvaluation results have been saved to:\n{outDir.joinpath('evaluation.txt').resolve()}")