#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Demo to test images
: Author - Xi Mo
: Institute - University of Kansas
: Date - 9/23/2021
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
from utils.network import GANet_dense_ga_accurate_small_link, add_fp16_bn_wrapper


parser = argparse.ArgumentParser("RGANet demo for suction dataset")
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by user, or use the latest ckpt in default folder './checkpoint'")
parser.add_argument("-d", "--dir", type = Path, default = r"samples",
                    help = r"Please specify the folder to read from/ save to")


# Please modify config to meet your requirements
CONFIG = {
	"FORMAT": ".jpg",
	"HAS_AMP": True
}


def check_ckpt_path(args) -> Path:
	if str(args.checkpoint) != "checkpoint":
		if not args.checkpoint.is_file():
			raise IOError(f"Designated checkpoint file does not exist:\n"
				      f"{args.checkpoint.resolve()}")
		ckptPath = args.checkpoint.resolve()
	else:
		ckptDir = Path.cwd().joinpath("checkpoint")
		if not args.checkpoint.is_dir():
			raise IOError(f"Default folder 'checkpoint' does not exist:\n"
				      f"{args.checkpoint.resolve()}")
		fileList = sorted(ckptDir.glob("*.pt"), reverse=True,
				  key=lambda item: item.stat().st_ctime)
		if len(fileList) == 0:
			raise IOError(f"Cannot find any checkpoint files in:\n"
				      f"{ckptDir.resolve()}\n")
		else:
			ckptPath = fileList[0]
	return ckptPath


def get_img_list(args):
	if str(args.dir) != "samples":
		if not args.dir.is_dir():
			raise IOError(f"Demo sample folder '{args.dir.name}' does not exist:\n")
		imgDir = args.dir.resolve()
	else:
		imgDir = Path.cwd().joinpath("samples")
		imgDir.mkdir(exist_ok = True)
	fileList = imgDir.rglob("*" + CONFIG["FORMAT"])
	return list(fileList)


def read_image(path: Path) -> torch.Tensor:
	imgData = Image.open(path).convert("RGB")
	w, h = imgData.size
	imgData = transforms.Compose([
						transforms.ToTensor(),
			    			transforms.Resize((480, 640),
								  interpolation = transforms.InterpolationMode.NEAREST)
						])(imgData)
	return imgData, (h, w)


def main():
	args = parser.parse_args()
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	ckptPath = check_ckpt_path(args)
	imgList = get_img_list(args)
	if len(imgList) == 0: return
	rganet = GANet_dense_ga_accurate_small_link(k=15)
	if CONFIG["HAS_AMP"]: add_fp16_bn_wrapper(rganet)
	checkpoint = torch.load(ckptPath)
	rganet.load_state_dict(checkpoint['model_state_dict'])
	print(f"\nCheckpoint loaded:\n{ckptPath.absolute()}")
	rganet.eval()
	rganet.to(device)
	with torch.no_grad():
		print(f"\nSaving predictions to:\n{args.dir.absolute()}\n")
		for idx, imgPath in enumerate(imgList):
			if imgPath.stem[-7:] == "predict": continue
			name = imgPath.stem + "_predict" + CONFIG["FORMAT"]
			img, size = read_image(imgPath)
			pred = rganet(img.unsqueeze(0).to(device))
			pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
			pred = transforms.Resize(size, transforms.InterpolationMode.NEAREST)(pred).squeeze(0)
			pred = pred.detach().div(2.0)
			save_image(pred.cpu(), args.dir.joinpath(name))
			print(f"Image {imgPath.name} saved, {len(imgList) - idx - 1} remaining .......")
		print(f"\nPredictions have been saved to:\n{args.dir.absolute()}")


if __name__ == "__main__":
	main()
