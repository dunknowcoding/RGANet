#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Test speed of different models
: standalone segmentation using torch in-built models - fcn, deeplabv3, lr-aspp
: Author - Xi Mo
: Institute - University of Kansas
: Date - 5/13/2021 last updated 5/15/2021
: Model Reference:
    https://pytorch.org/vision/stable/models.html#semantic-segmentation
: HowTo:
    0) This script adopts functoins from RGANet codes, can be execuated independently.
    Requirments: pytorch >= 1.0.0, python >= 3.6, numpy
    create a folder "checkpoint" at the same level of folder "dataset", then download correct
    checkpoint ot the created folder, then run this script.
    1) To specify parameters, refer to CONFIG and parser for details.
    2) You can copy and rename this script to run several different models at a time,
    to do this, you must specify correct gpu using '-gpu' parameter (default: 0)

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

parser = argparse.ArgumentParser(description="Arguments for training, validation and testing")

parser.add_argument("-gpu", type = int, default = 0,
                    help = "Designate GPU # for trainig and testing")

# Accomodation to suction dataset
parser.add_argument("-i", "--image", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/color-input",
                    help = "Directory to training images")
parser.add_argument("-l", "--label", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/label",
                    help = "Directory to training annotations")
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by users")

CONFIG = {
    "MODEL": "lr-aspp",            # choose between "fcn", "deeplab", "lr-aspp"
    "BACKBONE": "resnet50",        # backbone "resnet50", "resnet101" for fcn
                                   # backbone "resnet50", "resnet101", "mobilenet" for "deeplab"
                                   # backbone "mobilenet" for "lr-aspp" (this will ignore backbone
                                   # setting)

    "NUM_CLS": 3,               # number of classes
    "INT_CLS": (255, 0, 128),   # raw label intensity levels to differentiate classes

    # Testing
    "TEST_BATCH": 20,           # batchsize for testing
    "TEST_TIME": 1,             # show runtime stats done running certain number of testing batches
    "TEST_WORKERS": 0,          # set number of workers to run testing batches
    "TEST_PIN": True,           # set to True if memory is pinned for testing batches
    "TEST_RUNTIME": True,       # False to disable runtime test
    }

class SuctionDataset(torch.utils.data.Dataset):
    def __init__(self, imgDir, labelDir, splitDir=None, mode="test", applyTrans=False, sameTrans=True):
        super(SuctionDataset).__init__()
        assert len(CONFIG["INT_CLS"]) > 1, "Must be more than 1 class"
        assert len(CONFIG["INT_CLS"]) == CONFIG["NUM_CLS"], "Number of class does not match intensity levels"
        self.applyTran = applyTrans
        self.sameTrans = sameTrans
        self.mode = mode
        if splitDir and labelDir:
            self.img = self.read_split_images(imgDir, splitDir, ".png", 1)
            self.imgLen = len(self.img)
            assert self.imgLen, "Empty dataset, please check directory"
            self.nameList = list(self.img.keys())
            self.W, self.H = self.img[self.nameList[0]].size
            self.label = self.read_split_images(labelDir, splitDir, ".png", 0)
        else:
            raise IOError("Must specify training split file and annotation directory")

    # get one pair of samples
    def __getitem__(self, idx):
        imgName = self.nameList[idx]
        img, label = self.img[imgName], self.label[imgName]
        # necesary transformation
        operate = transforms.Compose([transforms.ToTensor(), self._transform_pad_image()])
        img = operate(img)
        label = self._convert_img_to_uint8_tensor(label)
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
        intLevels = CONFIG["INT_CLS"]

        for idx, val in enumerate(intLevels):
            save.append(np.where(dummy == val))
        for idx, val in enumerate(save):
            dummy[val] = idx
        dummy = torch.tensor(dummy, dtype = torch.uint8)
        dummy = self._transform_pad_image()(dummy)
        return dummy

# Helper to select model
def model_paser():
    if CONFIG["MODEL"] == "fcn":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(pretrained=False, progress=True,
                                 aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import fcn_resnet101
            model = fcn_resnet101(pretrained=False, progress=True,
                                  aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for FCN.")
    elif CONFIG["MODEL"] == "deeplab":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=False, progress=True,
                                       aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import deeplabv3_resnet101
            model = deeplabv3_resnet101(pretrained=False, progress=True,
                                        aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "mobilenet":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
            model = deeplabv3_mobilenet_v3_large(pretrained=False,
                                                 progress=True,
                                                 aux_loss=False,
                                                 num_classes=CONFIG["NUM_CLS"])
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for DeepLabv3.")
    elif CONFIG["MODEL"] == "lr-aspp":
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large
        CONFIG["BACKBONE"] = "mobilenet"
        model = lraspp_mobilenet_v3_large(pretrained=False, progress=True,
                                          aux_loss=False, num_classes=CONFIG["NUM_CLS"])
    else:
        raise NameError(f"Unsupported network \"{CONFIG['MODEL']}\" for now, I'd much appreciate "
                        f"if you customize more state-of-the-arts architectures.")
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    assert isinstance(args.gpu, int), "invalid numerical format to designate GPU"
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")

    torch.autograd.set_detect_anomaly(True)

    ''' Test Only'''

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
            gStartTime = time.time()
            for idx, data in enumerate(testSet):
                if idx == 0: bStartTime = time.time()
                img = data[0].to(device)
                labelOut = model(img)["out"][0]
                _ = torch.softmax(labelOut, dim=1)
                if (idx + 1) % CONFIG["TEST_TIME"] == 0:
                    bEndTime = time.time()
                    bRunTime = (bEndTime - bStartTime) * 1e3 / (CONFIG["TEST_TIME"] * CONFIG["TEST_BATCH"])
                    bStartTime = time.time()
                    print("batch: %4d/%d, average_runtime (%d batches): %6fms per image"
                          % (idx + 1, totalBatch, CONFIG["TEST_TIME"], bRunTime))

            gEndTime = time.time()
            gRunTime = (gEndTime - gStartTime) * 1e3 / (totalBatch * CONFIG["TEST_BATCH"])
            print("\n======================= %s Test Done =======================" %(CONFIG["MODEL"]))
            print("Average (%d images in total): %6fms\n" % (len(testData), gRunTime))
