#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Get adaptive weights for loss functions
: Author - Xi Mo
: Institute - University of Kansas
: revised Date - 12/24/2021
: HowTo: Set 'cfg' according to the dataset, only labels are processed
"""

from pathlib import Path
from PIL import Image
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

cfg = {
	"LAB": Path(r"../dataset/suction-based-grasping-dataset/data/label"),  # label directory or label root directory
	"PFIX": ".png",  # image post fix to match, suction dataset
	"GREY": 0,  # True if labels are greyscale images
	"GREYINT": (255, 0, 128),  # intensities for each class specified by index, for greyscale images only
	"IGNORE": 0,  # True classify unprocessed image regions as len(COLRINT) - th class
	"IMPECT": 100,  # for reversed, set the importance of non-exisiting category, higher the more important
	"METHOD": "all",  # the way proportions are handled, selected from "reverse", "raw", "focal", "all"
	"NORM": 1,  # True if the weights are normalized
	"COE": 1,  # coefficient for 'focal' choice, integer/list to specify coefficients for all class
	"GAMMA": 1,  # gamma for 'focal' choice, gamma > 0
}


def get_proportion(p: Path) -> np.ndarray:
	if cfg["GREY"]:
		prop = []
		img = np.array(Image.open(p).convert('L'))
		H, W = img.shape
		for v in cfg["GREYINT"]: prop.append(np.sum(img == v))
	else:
		prop = [0] * len(cfg["COLRINT"])
		img = np.array(Image.open(p))
		H, W, _ = img.shape
		for c, patch in enumerate(cfg["COLRINT"].items()):
			_, v = patch
			prop[c] = np.sum((img[:, :, 0] == v[0]) & (img[:, :, 1] == v[1]) & (img[:, :, 2] == v[2]))

	prop = [v / (H * W) for v in prop]
	if cfg["IGNORE"]: prop.append(1.0 - sum(prop))
	if cfg["METHOD"] == "reverse":
		for idx, v in enumerate(prop): prop[idx] = cfg["IMPECT"] if not v else 1 / v

	elif cfg["METHOD"] == "focal":
		if cfg["GAMMA"] <= 0: cfg["GAMMA"] = 1
		if type(cfg["COE"]) == list:
			assert len(cfg["COE"]) == len(cfg["GREYINT"]), "coefficients does not match classses"
			for idx, v in enumerate(cfg["COE"]): prop[idx] = v * pow(1 - prop[idx], cfg["GAMMA"])
		else:
			for idx, v in enumerate(prop): prop[idx] = cfg["COE"] * pow(1 - v, cfg["GAMMA"])

	if cfg["NORM"]:
		total = sum(prop)
		prop = [ v / total for v in prop]

	return np.array(prop)


def get_all_proportion(p: Path) -> np.ndarray:
	if cfg["GREY"]:
		prop = []
		img = np.array(Image.open(p).convert('L'))
		H, W = img.shape
		for v in cfg["GREYINT"]: prop.append(np.sum(img == v))
	else:
		prop = [0] * len(cfg["COLRINT"])
		img = np.array(Image.open(p))
		H, W, _ = img.shape
		for c, patch in enumerate(cfg["COLRINT"].items()):
			_, v = patch
			prop[c] = np.sum((img[:, :, 0] == v[0]) & (img[:, :, 1] == v[1]) & (img[:, :, 2] == v[2]))

	prop = [v / (H * W) for v in prop]
	if cfg["IGNORE"]: prop.append(1.0 - sum(prop))
	prop1, prop2 = prop.copy(), prop.copy()
	for idx, v in enumerate(prop): prop1[idx] = cfg["IMPECT"] if not v else 1 / v
	if cfg["GAMMA"] <= 0: cfg["GAMMA"] = 1
	if type(cfg["COE"]) == list:
		assert len(cfg["COE"]) == len(cfg["GREYINT"]), "coefficients does not match classses"
		for idx, v in enumerate(cfg["COE"]): prop2[idx] = v * pow(1 - prop[idx], cfg["GAMMA"])
	else:
		for idx, v in enumerate(prop): prop2[idx] = cfg["COE"] * pow(1 - v, cfg["GAMMA"])

	if cfg["NORM"]:
		total = sum(prop)
		prop = [ v / total for v in prop]
		total = sum(prop1)
		prop1 = [ v / total for v in prop1]
		total = sum(prop2)
		prop2 = [ v / total for v in prop2]

	return np.array(prop), np.array(prop1), np.array(prop2)


if __name__ == '__main__':
	# read images
	imgList = sorted(cfg["LAB"].rglob("*" + cfg["PFIX"]))
	length = len(list(imgList))
	if cfg["METHOD"] != "all":
		pbar = tqdm(total=length, unit="files")
		for index, p in enumerate(imgList):
			pbar.set_postfix(image=str(p.resolve()))
			if index == 0:
				prop = get_proportion(p)
			else:
				prop += get_proportion(p)

			pbar.update(1)
		pbar.close()
	else:
		pbar = tqdm(total = length, unit = "files")
		for index, p in enumerate(imgList):
			pbar.set_postfix(image = str(p.resolve()))
			if index == 0:
				raw, reverse, focal = get_all_proportion(p)
			else:
				raw_n, reverse_n, focal_n = get_all_proportion(p)
				raw += raw_n
				reverse += reverse_n
				focal += focal_n
			pbar.update(1)

		pbar.close()
	# print results
	if cfg["METHOD"] != "all":
		prop /= (index + 1)
		prop = [round(v, 3) for v in prop]
		print(f"\nThe weights calculated by {cfg['METHOD']} is:\n{list(prop)}")
	else:
		raw  /= (index + 1)
		raw = [round(v, 3) for v in raw]
		reverse /= (index + 1)
		reverse = [round(v, 3) for v in reverse]
		focal /= (index + 1)
		focal = [round(v, 3) for v in focal]
		print(f"\nThe weights calculated by 'raw' is:\n{list(raw)}")
		print(f"\nThe weights calculated by 'reverse' is:\n{list(reverse)}")
		print(f"\nThe weights calculated by 'focal' is:\n{list(focal)}")
