#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Validation for GANet
: Author - Xi Mo
: Institute - University of Kansas
: Date - revised on 12/24/2021
"""

import numpy as np
import math
import torch
from pathlib import Path
from utils.configuration import CONFIG
from PIL import Image


''' 
    Options for save metrics to disk
    
    "all":          Write all metrics
    "iou":          Jaccard index per cls
    "acc":          correct pixels per cls
    "dice":         F1-score per cls
    "precision":    Precisoin per cls
    "recall":       Recall per cls
    "mgrid":        mean grid per cls
    "roc":          TPR and FPR for calculating ROC per cls
    "mcc":          Phi coefficient per cls
    
'''
# Metric for Offline/Online CPU mode
class Metrics:
	def __init__(self, label, gt, cls=CONFIG["NUM_CLS"] - 1, one_hot=False):
		assert gt.ndim == 2, "groundtruth must be grayscale image"
		# one_hot label to unary
		if one_hot:
			assert label.ndim == 3, "label must be 3-dimensional for one-hot encoding"
			label = np.argmax(label, axis=2)
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

	# Jaccard: TP/(FP+TP+FN)
	def IOU(self) -> np.float32:
		UN = self.TP_FN + self.TP_FP
		if UN == 0: return 1.0
		return np.float32(self.TP / (UN - self.TP + 1e-31))

	# acc: TP+TN/(TP+FP+TN+FN)
	def ACC(self) -> np.float32:
		accuracy = (self.TP + self.TN) / (self.H * self.W)
		return np.float32(accuracy)

	# dice: Sørensen–Dice coefficient 1/(1/precision + 1/recall)
	# precision, recall(aka TPR), FPR
	def DICE(self) -> [np.float32]:
		if self.TP == self.FN == self.FP == 0: return 1.0
		precision = np.float32(self.TP / (self.TP + self.FP + 1e-31))
		recall = np.float32(self.TP / (self.TP + self.FN + 1e-31))
		dice = np.float32(2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-31))
		return dice

	# precision
	def PRECISION(self) -> np.float32:
		if self.TP == self.FN == self.FP == 0: return 1.0
		return np.float32(self.TP / (self.TP + self.FP + 1e-31))

	# recall
	def RECALL(self) -> np.float32:
		if self.TP == self.FN == self.FP == 0: return 1.0
		return np.float32(self.TP / (self.TP + self.FN + 1e-31))

	# TPR and FPR for ROC curve
	def ROC(self) -> [np.float32]:
		if self.TP == self.FN == 0 and self.FP == self.TN == 0:
			return [1.0, 1.0]

		if not (self.TP == self.FN == 0) and self.FP == self.TN == 0:
			tpr = np.float32(self.TP / (self.TP + self.FN))
			return [tpr, 1.0]

		if not (self.FP == self.TN == 0) and self.TP == self.FN == 0:
			fpr = np.float32(self.FP / (self.FP + self.TN))
			return [1.0, fpr]

		tpr = np.float32(self.TP / (self.TP + self.FN))
		fpr = np.float32(self.FP / (self.FP + self.TN))
		return [tpr, fpr]

	# mcc: Matthews correlation coefficient (Phi coefficient)
	def MCC(self) -> np.float32:
		if self.TP == self.FN == self.FP == 0: return 1.0
		N = self.TN + self.TP + self.FN + self.FP
		S = (self.TP + self.FN) / N
		P = (self.TP + self.FP) / N
		if S == 0 or P == 0: return -1.0
		if S == 1 or P == 1: return 0.0
		return np.float32((self.TP / N - S * P) / math.sqrt(P * S * (1 - S) * (1 - P)))

	# mgrid: a mean grid F_beta score metric, beta > 1 (=2) or beta < 1 (=0.5)
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
				Fbeta = np.float32((1 + beta ** 2) * TP / (beta ** 2 * (TP + FN) + TP + FP + 1e-31))
				runMean += curve(Fbeta)

		return np.float32(runMean / cnt) if cnt else 1.0


	'''
	    options:
	    'all': save all evalutaion metrics to disk
	    otherwise: specify the metric to be saved, refer to line 19
	'''
	# evalute and save results to disk
	def save_to_disk(self, name: str, path: Path, option="all",
			 interval=(12, 12), beta=5, shift=(0.525, 0.5)):
		path = path.joinpath("evaluation.txt")

		if option == "all":
			with open(path, "a+") as f:
				iou, acc, dice = 100 * self.IOU(), 100 * self.ACC(), 100 * self.DICE()
				precsion, recall = 100 * self.PRECISION(), 100 * self.RECALL()
				tpr, fpr = self.ROC()
				tpr *= 100
				fpr *= 100
				mcc = 100 * self.MCC()
				mgrid = 100 * self.MGRID(interval, beta, shift)
				f.write(f"{name} iou:{iou:.2f} acc:{acc:.2f} precision:{precsion:.2f} "
					f"recall:{recall:.2f} dice:{dice:.2f} mgrid:{mgrid:.2f} "
					f"tpr:{tpr:.2f} fpr:{fpr:.2f} mcc:{mcc:.2f}\n")
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
				dice = 100 * self.DICE()
				f.write(f"{name:s} dice:{dice:.2f}\n")
			return
		# write mgrid only
		if option == "mgrid":
			with open(path, "a+") as f:
				mgrid = 100 * self.MGRID(interval, beta, shift)
				f.write(f"{name:s} mgrid:{mgrid:.2f}\n")
			return
		# write precision only
		if option == "precision":
			with open(path, "a+") as f:
				precision = 100 * self.PRECISION()
				f.write(f"{name:s} precision:{precision:.2f}\n")
			return
		# write recall only
		if option == "recall":
			with open(path, "a+") as f:
				recall = 100 * self.RECALL()
				f.write(f"{name:s} precision:{recall:.2f}\n")
			return
		# write roc only
		if option == "roc":
			with open(path, "a+") as f:
				tpr, fpr = 100 * self.ROC()
				f.write(f"{name:s} tpr:{tpr:.2f} fpr:{fpr:.2f}\n")
			return
		# write mcc only
		if option == "mcc":
			with open(path, "a+") as f:
				mcc = 100 * self.MCC()
				f.write(f"{name:s} mcc:{mcc:.2f}\n")
			return


	'''
	
	    Return a dict of metrics for further processing, options:
	    "all" or []:    all metrics
	    [metrics]:      selected metrics by names, refer to line 19('roc' -> 'tpr' and 'fpr')
	    
	'''
	# generate evaluations on-the-fly
	def values(self, options="all", interval=(12, 12), beta=0.5, shift=(0.525, 0.5)):
		varDict = {"iou": None, "acc": None, "dice": None,
			   "precision": None, "recall": None, "mgrid": None,
			   "tpr": None, "fpr": None, "mcc": None}
		if options == "all" or options == []:
			options = ["iou", "acc", "dice", "precision",
				   "recall", "mgrid", "tpr", "fpr", "mcc"]

		for metric in options:
			if metric == "iou": varDict[metric] = self.IOU()
			if metric == "acc": varDict[metric] = self.ACC()
			if metric == "dice": varDict[metric] = self.DICE()
			if metric == "mgrid": varDict[metric] = self.MGRID(interval, beta, shift)
			if metric == "precision": varDict[metric] = self.PRECISION()
			if metric == "recall": varDict[metric] = self.RECALL()
			if metric == "tpr": varDict[metric] = self.ROC()[0]
			if metric == "fpr": varDict[metric] = self.ROC()[1]
			if metric == "mcc": varDict[metric] = self.MCC()

		return varDict


''' fast GPU implementation of evaluatiing a batch online '''

class Metrics_gpu:
	def __init__(self, label, gt, one_hot=False):
		assert gt.shape[0] == label.shape[0], "Batchsize does not match"
		assert len(gt.shape) == 3, "groundtruth - [bathsize, h, w]"
		# one_hot label to unary
		if one_hot:
			assert len(label.shape) == 4, "label must be 4-dimensional for one-hot encoding"
			label = torch.argmax(label, dim=1)
		else:
			assert len(label.shape) == 3, "label must be 3-dimensional"

		self.H, self.W = CONFIG["SIZE"]
		self.bSize = gt.shape[0]
		cls = CONFIG["NUM_CLS"] - 1
		label_area, gt_area = torch.where(label == cls)[0], torch.where(gt == cls)[0]
		doubled = torch.add(label, gt)
		self.TP = len(torch.where(doubled == 2 * cls)[0])
		self.TP_FN = len(gt_area)
		self.TP_FP = len(label_area)
		self.FP = self.TP_FP - self.TP
		self.TN = self.H * self.W * self.bSize - self.TP_FN - self.TP_FP + self.TP
		self.FN = self.TP_FN - self.TP

	# Jaccard: TP/(FP+TP+FN)
	def IOU(self):
		UN = self.TP_FN + self.TP_FP
		if UN == 0: return 1.0
		return self.TP / (UN - self.TP + 1e-31)

	# acc: TP+TN/(TP+FP+TN+FN)
	def ACC(self):
		return (self.TP + self.TN) / (self.H * self.W * self.bSize)

	# dice: Sørensen–Dice coefficient 1/(1/precision + 1/recall)
	# precision, recall(aka TPR), FPR
	def DICE(self):
		if self.TP == self.FN == self.FP == 0: return 1.0
		return 2 * self.TP / (2 * self.TP + self.FP + self.FN + 1e-31)

	# precision
	def PRECISION(self):
		if self.TP == self.FN == self.FP == 0: return 1.0
		return self.TP / (self.TP + self.FP + 1e-31)

	# recall
	def RECALL(self):
		if self.TP == self.FN == self.FP == 0: return 1.0
		return self.TP / (self.TP + self.FN + 1e-31)


''' save fast online report to disk '''

def save_report(path: Path, info: dict, ckptPath: Path):
	# write details
	with open(path, "a+") as f:
		f.write(f"\n\nCheckpoint folder:\n{ckptPath.absolute()}\n\n")
		for i, name in enumerate(info["ckpt"]):
			f.write(f"{name} "
				f"iou:{info['iou'][i]:.2f} "
				f"acc:{info['acc'][i]:.2f} "
				f"precision:{info['prec'][i]:.2f} "
				f"recall:{info['rec'][i]:.2f} "
				f"dice:{info['dice'][i]:.2f}\n")

		f.write("\n")
	# write statistics
	for key in info.keys():
		if key != "ckpt":
			with open(path, "a+") as f:
				max_ = max(info[key])
				i = info[key].index(max_)
				f.write(f"Highest {key}: {max_:.2f}\n"
					f"Checkpoint:{info['ckpt'][i]}\n"
					f"Metircs: iou-{info['iou'][i]:.2f}, "
					f"accuracy-{info['acc'][i]:.2f}, precision-{info['prec'][i]:.2f}, "
					f"recall-{info['rec'][i]:.2f}, dice-{info['dice'][i]:.2f}\n\n")

	return
