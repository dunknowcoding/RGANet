#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Dataloadar for robotic hand grasping and suction dataset
: Author - Xi Mo
: Institute - University of Kansas
: Date - revised on 12/24/2021
"""

import numpy as np
import random
import torch
import gc

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from utils.configuration import CONFIG


''' Sunction dataset dataloader '''

class SuctionGrasping(torch.utils.data.Dataset):
    def __init__(self, imgDir, labelDir, splitDir=None, mode="test", applyTrans=False, sameTrans=True):
        super(SuctionGrasping).__init__()
        assert len(CONFIG["INT_CLS"]) > 1, "Must be more than 1 class"
        assert len(CONFIG["INT_CLS"]) == CONFIG["NUM_CLS"], "Number of class does not match intensity levels"
        assert len(CONFIG["SIZE"]) == 2, "Invalid SIZE format"
        assert type(CONFIG["PAR_SFT"]) == tuple and len(CONFIG["PAR_SFT"]) == 2, "Invalid SHIFT parameters"
        assert type(CONFIG["PAR_NORM"]) == dict, "Mean and std must be presented in a dict"
        self.applyTrans = applyTrans
        self.sameTrans = sameTrans
        self.mode = mode
        if mode in ["train", "test", "validate"]:
            if splitDir and labelDir:
                self.img = self.read_split_images(imgDir, splitDir, CONFIG["POSTFIX"], 1)
                self.imgLen = len(self.img)
                assert self.imgLen, "Empty dataset, please check directory"
                self.nameList = list(self.img.keys())
                self.W, self.H = self.img[self.nameList[0]].size
                self.label = self.read_split_images(labelDir, splitDir, CONFIG["POSTFIX"], 0)

        torch.cuda.empty_cache()

    def __getitem__(self, idx):
        imgName = self.nameList[idx]
        img, label  = self.img[imgName], self.label[imgName]
        if self.applyTrans:
            img = transforms.Compose([transforms.ToTensor(), self.transform_resize_image()])(img)
            img, label = self.img_normalize(img), self._convert_img_to_uint8_tensor(label)
            img = self.img_random_color_jitter(img)
            img = self.img_random_blur(img)
            img, label = self.img_random_flip(img, label)
            img, label = self.img_random_shift_rotate(img, label)
        else:
            img = transforms.Compose([transforms.ToTensor(), self.transform_resize_image()])(img)
            img, label = self.img_normalize(img), self._convert_img_to_uint8_tensor(label)
        return img, label, imgName

    def __len__(self):
        return self.imgLen

    # read names/directories from text files
    @classmethod
    def read_image_id(cls, filePath: Path, postFix: str) -> [str]:
        assert filePath.is_file(), f"Invalid file path:\n{filePath.resolve()}"
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
            assert imgPath.is_file(), f"Invalid image path: \n{imgPath.resolve()}"
            img = Image.open(imgPath).convert('RGB')
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

        dummy = torch.unsqueeze(torch.as_tensor(dummy, dtype = torch.uint8), 0)
        dummy = self.transform_resize_image(interpoloation = transforms.InterpolationMode.NEAREST)(dummy)
        return dummy.squeeze(0)

    # one-hot encoder for int64 label tensor
    @staticmethod
    def one_hot_encoder(label: torch.Tensor) -> torch.Tensor:
        assert len(label.shape) == 3, r"Length of the tensor must be [batchSize, H, W]"
        label = label.to(torch.int64)
        dummy = torch.nn.functional.one_hot(label, CONFIG["NUM_CLS"])
        return dummy.permute(0, 3, 1, 2).to(torch.float32)

    # write testing results to disk
    @staticmethod
    def save_results(batch: torch.Tensor, folder: Path, name: int, postfix=".png", bgr=False, pred=False):
        assert len(batch.shape) in [3, 4], r"Must be 4-dim/3-dim tensor for color/greyscale images"
        data = batch.cpu().clone()
        for idx in range(data.shape[0]):
            file = str(name + idx) + postfix
            filepath = folder.joinpath(file)
            if len(data.shape) == 4:
                img = data[idx, :, :, :]
                # denormalization for samples, norm params MUST match those during training
                if not pred and not CONFIG["DENORM"] and CONFIG["HAS_NORM"]:
                    img = img.view(3, -1) * torch.tensor(CONFIG["PAR_NORM"]["std"]).unsqueeze(1)
                    img += torch.tensor(CONFIG["PAR_NORM"]["mean"]).unsqueeze(1)
                    img = img.reshape(3, *CONFIG["SIZE"])
                # for predication
                if pred:
                    img = img / (img.max() + 1e-31)
                    # RGB to BGR display, for prediction
                    if bgr:
                        index = [2, 1, 0]
                        img = img[index]
            elif len(data.shape) == 3:
                img = data[idx, :, :]
                img = img.to(torch.float32)
                img = img / (CONFIG["NUM_CLS"] - 1 + 1e-31)
                img = img.unsqueeze(0)

            img = SuctionGrasping.transform_resize_image((480, 640), transforms.InterpolationMode.NEAREST)(img)
            img = img.squeeze()
            save_image(img, filepath)
        return

    # padding
    def _transform_pad_image(self):
        H, W = CONFIG["SIZE"]
        dH, dW = max(0, H-self.H), max(0, W-self.W)
        padding = [dW//2, dH//2, dW-dW//2, dH-dH//2]
        return transforms.Pad(padding=padding, padding_mode='constant')

    # resize
    @staticmethod
    def transform_resize_image(size = CONFIG["SIZE"], interpoloation = transforms.InterpolationMode.BILINEAR):
        return transforms.Resize(size, interpolation = interpoloation)

    # shift and rotation
    def img_random_shift_rotate(self, img: torch.Tensor, label: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        if CONFIG["SHIFT"] or CONFIG["ROTATE"]:
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
    @classmethod
    def img_random_color_jitter(cls, img: torch.Tensor) -> torch.Tensor:
        if random.random() < CONFIG["P_JITTER"]:
            operator = transforms.ColorJitter(CONFIG["BRIGHTNESS"], CONFIG["CONTRAST"], CONFIG["HUE"])
            img = operator(img)
        return img

    # normalization
    @classmethod
    def img_normalize(cls, img: torch.Tensor) -> torch.Tensor:
        if CONFIG["HAS_NORM"]:
            operator = transforms.Normalize(CONFIG["PAR_NORM"]["mean"], CONFIG["PAR_NORM"]["std"])
            img = operator(img)
        return img

    # Gaussian blur
    @classmethod
    def img_random_blur(cls, img: torch.Tensor):
        if CONFIG["BLUR"]:
            if random.random() < CONFIG["P_BLUR"]:
                operator = transforms.GaussianBlur(CONFIG["PAR_BLUR"]["kernel"], CONFIG["PAR_BLUR"]["sigma"])
                img = operator(img)
        return img
