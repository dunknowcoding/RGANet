#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Training and testing
: Author - Xi Mo
: Institute - University of Kansas
: Date - revised on 12/24/2021
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dataset

from pathlib import Path
from matplotlib import pyplot as plt

from utils.configuration import parser, CONFIG
from utils.dataLoader import SuctionGrasping
from utils.validate import Metrics, Metrics_gpu, save_report
from utils.ga_crf import read_image_from_disk, save_image_to_disk, trans_img_to_cls
from utils.network import KLD, GANet_dense_ga_accurate_small_link, focalLoss, \
						save_model, optimizer, add_fp16_bn_wrapper


# AMP import
VERSION = float(".".join(torch.__version__.split(".")[:2]))
if CONFIG["HAS_AMP"]:
	if VERSION < 1.6:
		try:
			import apex
		except ImportError:
			print(r"Package Apex missing, please refer to https://github.com/NVIDIA/apex for installation")
		else:
			from apex import amp
	else:
		from torch.cuda.amp import autocast as autocast


def test(args, device):
	if CONFIG["DENORM"]: CONFIG["HAS_NORM"] = False
	# checkpoint filepath check
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
	if CONFIG["DATASET"] == "suction":
		testSplitPath = args.image.parent.joinpath("test-split.txt")
		if not testSplitPath.is_file():
			raise IOError(f"Test-split file does not exist, please download the dataset first:\n"
				      f"{trainSplitPath}")
		testData = SuctionGrasping(args.image, args.label, testSplitPath,
					   mode="test", applyTrans=False, sameTrans=False)
	else:
		raise ValueError("Unsupported dataset in this release")

	testSet = dataset.DataLoader(dataset=testData,
				  batch_size=CONFIG["TEST_BATCH"],
				  shuffle=False,
				  num_workers=CONFIG["TEST_WORKERS"],
				  pin_memory=CONFIG["TEST_PIN"],
				  drop_last=False)
	print(f"{CONFIG['DATASET']} dataset loaded.\n")
	# RGANet testing
	if CONFIG["DATASET"] == "suction":
		ganet = GANet_dense_ga_accurate_small_link(k = 15)
	else:
		raise NameError(f"Unspported dataset '{CONFIG['DATASET']}' in this version")

	checkpoint = torch.load(ckptPath)
	if CONFIG["HAS_AMP"]: add_fp16_bn_wrapper(ganet)
	ganet.load_state_dict(checkpoint['model_state_dict'])
	print(f"\nCheckpoint loaded for testing RGANet:\n{ckptPath.absolute()}")
	ganet.eval()
	ganet.to(device)
	assert CONFIG["TEST_BATCH"] >= 1, "Test batchsize must be a positive integer"
	CONFIG["TEST_BATCH"] = int(CONFIG["TEST_BATCH"])
	totalBatch = np.ceil(len(testData) / CONFIG["TEST_BATCH"])
	with torch.no_grad():
		# get accurate inference time estimation
		if CONFIG["TEST_RUNTIME"]:
			if CONFIG["TEST_TIME"] < 1: CONFIG["TEST_TIME"] = 1
			if CONFIG["TEST_MUL"] < 1: CONFIG["TEST_MUL"] = 1
			tailCount = len(testData) % CONFIG["TEST_BATCH"]
			totalTime = 0
			for i in range(CONFIG["TEST_MUL"]):
				print(f"\nFold {i + 1} of {CONFIG['TEST_MUL']}:\n")
				for idx, data in enumerate(testSet):
					img = data[0].to(device)
					torch.cuda.synchronize()
					startTime = time.time()
					_ = ganet(img)
					torch.cuda.synchronize()
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
		# set default results directory
		if str(args.dir) != "results":
			if not args.dir.is_dir():
				raise IOError(f"Invalid sample folder:\n{args.dir.resolve()}")
		else:
			args.dir = args.dir.with_name("results" + "_acrt_" + CONFIG["DATASET"])
		# save results if required
		if CONFIG["TEST_SAVE"]:
			print(f"\nNow saving test results to:\n{args.dir.absolute()}\n")
			if CONFIG["DATASET"] == "suction":
				labelDir = args.dir.joinpath("annotations")
				labelDir.mkdir(exist_ok=True, parents=True)
				resultDir = args.dir.joinpath("output")
				resultDir.mkdir(exist_ok=True, parents=True)
				imgDir = args.dir.joinpath("images")
				imgDir.mkdir(exist_ok=True, parents=True)

			imgCnt = 1
			for idx, data in enumerate(testSet):
				img = data[0].to(device)
				labelOut = ganet(img)
				labelOut = torch.softmax(labelOut, dim=1)
				if CONFIG["DATASET"] == "suction":
					SuctionGrasping.save_results(data[0], imgDir, imgCnt, postfix=CONFIG["POSTFIX"])
					SuctionGrasping.save_results(data[1], labelDir, imgCnt, postfix=CONFIG["POSTFIX"])
					SuctionGrasping.save_results(labelOut, resultDir, imgCnt, postfix=CONFIG["POSTFIX"],
								     bgr=CONFIG["TEST_BGR"], pred=True)

				imgCnt += CONFIG["TEST_BATCH"]
				if (idx + 1) % CONFIG["TEST_TIME"] == 0:
					print(f"%4d/%d batches processed" % (idx + 1, totalBatch))

			print("\n==================== Test Results Saved ====================\n")


def train(args, device, version):
	assert 0 < CONFIG["SAVE_MODEL"] <= CONFIG["EPOCHS"], "Invalid interval of screenshot"
	# checkpoint filepath check
	if str(args.checkpoint) != "checkpoint":
		if not args.checkpoint.is_file():
			raise IOError(f"Designated checkpoint file does not exist:\n"
				      f"{args.checkpoint.resolve()}")
		ckptPath = args.checkpoint.resolve
	# Create checkpoint directory
	ckptDir = Path.cwd().joinpath("checkpoint")
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
	# load suction dataset
	if CONFIG["DATASET"] == "suction":
		print(f"\nLoading {CONFIG['DATASET']} dataset for training, please wait .......")
		trainSplitPath = args.image.parent.joinpath("train-split.txt")
		if not trainSplitPath.is_file():
			raise IOError(f"Train-split file does not exist, please download the dataset first:\n"
				      f"{trainSplitPath}")

		if CONFIG["AUGMENT"]:
			trainData = SuctionGrasping(args.image, args.label, trainSplitPath,
						    mode="train", applyTrans=True, sameTrans=True)
		else:
			trainData = SuctionGrasping(args.image, args.label, trainSplitPath, mode="train")

	trainSet = dataset.DataLoader(dataset=trainData,
				   batch_size=CONFIG["BATCHSIZE"],
				   shuffle=CONFIG["SHUFFLE"],
				   num_workers=CONFIG["NUM_WORKERS"],
				   pin_memory=CONFIG["PIN_MEMORY"],
				   drop_last=CONFIG["DROP_LAST"])
	print(f"{CONFIG['DATASET']} dataset loaded.\n")
	# RGANet training
	ganet = GANet_dense_ga_accurate_small_link(k = 15)
	# loss function
	if CONFIG["LOSS"] == "focal":  # FocalLoss
		lossFuncLabel = focalLoss(gamma=CONFIG["GAMMA"], weights=CONFIG["WEIGHT"],
					  reduction=CONFIG["REDUCT"])
	elif CONFIG["LOSS"] == "ce":  # Cross-Entropy Loss
		if CONFIG["WEIGHT"] is not None:
			weight = torch.FloatTensor(CONFIG["WEIGHT"]).to(device)
		else:
			weight = None
		if CONFIG["DATASET"] == "suction":
			lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight, reduction=CONFIG["REDUCT"])
		elif CONFIG["DATASET"] in ["cityscape", "camvid"]:
			if CONFIG["IGNORE"]:
				from utils.dataLoader import ignored
				lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight,
									  ignore_index=ignored, reduction=CONFIG["REDUCT"])
			else:
				lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight, reduction=CONFIG["REDUCT"])
	# lossFuncLabel = torch.nn.CrossEntropyLoss(weight=weight, reduction=CONFIG["REDUCT"])
	elif CONFIG["LOSS"] == "bce":  # Binary Cross-Entropy Loss
		if CONFIG["WEIGHT"] is not None:
			weight = torch.FloatTensor(CONFIG["WEIGHT"]).to(device)
		else:
			weight = None

		lossFuncLabel = torch.nn.BCELoss(weight=weight, reduction=CONFIG["REDUCT"])
	elif CONFIG["LOSS"] == "huber":  # Huber Loss
		lossFuncLabel = nn.SmoothL1Loss(beta=CONFIG["BETA"], reduction=CONFIG["REDUCT"])
	elif CONFIG["LOSS"] == "poisson":  # Poisson Loss
		lossFuncLabel = nn.PoissonNLLLoss(log_input=False, reduction=CONFIG["REDUCT"],
						  eps=CONFIG["PEPS"])
	elif CONFIG["LOSS"] == "kld":  # KLD divergence Loss
		lossFuncLabel = KLD(reduction=CONFIG["REDUCT"])
	else:
		raise NameError(f"Unspported loss function type '{CONFIG['LOSS']}'.")

	lossFuncLabel = lossFuncLabel.to(device)
	optimizer_GA = optimizer(ganet)
	if CONFIG["HAS_AMP"]: add_fp16_bn_wrapper(ganet)
	ganet.to(device)
	ganet.train()
	# set FP16 training
	if CONFIG["HAS_AMP"]:
		if version < 1.6:
			assert CONFIG["AMP_LV"] in [0, 1, 2, 3], "Unrecognized AMP level, please check"
			ganet, optimizer_GA = amp.initialize(ganet, optimizer_GA,
							     opt_level="O" + str(CONFIG["AMP_LV"]))
			amp.register_float_function(torch, "sigmoid")
		else:
			scaler = torch.cuda.amp.GradScaler(enabled= True)
	# set multi-step learining rate
	if CONFIG["LR_STEP"] is not None:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_GA,
						milestones = CONFIG["LR_STEP"], gamma = CONFIG["LR_MUL"])
	# load checkpoint if restore is true
	if args.restore:
		if CONFIG["HAS_AMP"]:
			try:
				checkpoint = torch.load(ckptPath)
			except:
				print("Checkpoint does not match AMP level or current pytorch version")
			else:
				print(f"\nCheckpoint successfully loaded:\n{ckptPath}\n")

			if version < 1.6:
				amp.load_state_dict(checkpoint['amp'])
			else:
				scaler.load_state_dict(checkpoint["scaler"])
		# load scheduler
		if CONFIG["LR_STEP"] is not None:
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		# load others
		ganet.load_state_dict(checkpoint['model_state_dict'])
		optimizer_GA.load_state_dict(checkpoint['optimizer_state_dict'])
		lastEpoch = checkpoint['epoch']
		if lastEpoch == CONFIG["EPOCHS"]:
			print("WARNING: Previous training has been finished, initialize transfer training ...... \n")
			lastEpoch = 0
	else:
		lastEpoch = 0

	totalBatch = np.ceil(len(trainData) / CONFIG["BATCHSIZE"])
	for epoch in range(lastEpoch, CONFIG["EPOCHS"]):
		runLoss = 0.0
		for idx, data in enumerate(trainSet):
			torch.cuda.synchronize()
			start_time = time.time()
			img, label, _ = data
			if CONFIG["LOSS"] in ["ce", "focal"]:
				label = label.long()
			elif CONFIG["LOSS"] in ["bce", "huber", "poisson", "kld"]:
				label = SuctionGrasping.one_hot_encoder(label)

			label = label.to(device)
			img = img.to(device)
			# train one iteration
			optimizer_GA.zero_grad()

			if CONFIG["HAS_AMP"]:
				if version < 1.6:
					output = ganet(img)
					loss = lossFuncLabel(output, label)
					with amp.scale_loss(loss, optimizer_GA) as scaled_loss:
						scaled_loss.backward()

					optimizer_GA.step()
				else:
					with autocast():
						output = ganet(img)
						loss = lossFuncLabel(output, label)

					scaler.scale(loss).backward()
					scaler.step(optimizer_GA)
					scaler.update()
			else:
				output = ganet(img)
				loss = lossFuncLabel(output, label)
				loss.backward()
				optimizer_GA.step()

			runLoss += loss.detach().item()
			torch.cuda.synchronize()
			runtime = (time.time() - start_time) * 1e3
			# print info
			if idx % CONFIG["SHOW_LOSS"] == CONFIG["SHOW_LOSS"] - 1:
				# Simple evaluation for a batch average, class_id: NUM_CLS-1
				if CONFIG["VAL_BATCH"]:
					with torch.no_grad():
						pred = torch.argmax(output.detach(), dim=1)
						labs = label.detach()
						if len(labs.shape) == 4: labs = torch.argmax(labs, dim=1)
						if CONFIG["DATASET"] == "suction":
							# fast evaluation
							TP_FP = len(torch.where(pred == 2)[0])
							TP_FN = len(torch.where(labs == 2)[0])
							TP = len(torch.where(torch.add(pred, labs) == 4)[0])

						IU = float(torch.div(TP, TP_FP + TP_FN - TP + 1e-31)) * 100
						precision = float(torch.div(TP, TP_FP + 1e-31)) * 100
						recall = float(torch.div(TP, TP_FN + 1e-31)) * 100

					averLoss = runLoss / CONFIG["SHOW_LOSS"]
					if CONFIG["LR_STEP"]:
						print("Epoch: %2d -> iters: %4d/%d | loss: %.5f | runtime: %4.3f ms/iter | "
						      "lr: %.6f | IU-%.1f%%  P-%.1f%% R-%.1f%%"
						      % (epoch + 1, idx + 1, totalBatch, averLoss, runtime,
							 scheduler.get_last_lr()[0], IU, precision, recall))
					else:
						print("Epoch: %2d -> iters: %4d/%d | loss: %.5f | runtime: %4.3f ms/iter | "
						      "IU-%.1f%% P->%.1f%% R->%.2f%%"
						      % (epoch + 1, idx + 1, totalBatch, averLoss, runtime, IU, precision, recall))
				else:
					averLoss = runLoss / CONFIG["SHOW_LOSS"]
					if CONFIG["LR_STEP"]:
						print("Epoch: %2d -> iters: %4d/%d | loss: %.5f | runtime: %4.3f ms/iter | lr: %.6f"
						      % (epoch + 1, idx + 1, totalBatch, averLoss, runtime, scheduler.get_last_lr()[0]))
					else:
						print("Epoch: %2d -> iters: %4d/%d | loss: %.5f | runtime: %4.3f ms/iter"
						      % (epoch + 1, idx + 1, totalBatch, averLoss, runtime))

				runLoss = 0.0

		if CONFIG["LR_STEP"]: scheduler.step()
		# save checkpoint
		if epoch % CONFIG["SAVE_MODEL"] == 0:
			if CONFIG["HAS_AMP"]:
				if version < 1.6:
					if CONFIG["LR_STEP"]:
						save_model(ckptDir, ganet, epoch + 1, optimizer_GA, amp, scheduler, version)
					else:
						save_model(ckptDir, ganet, epoch + 1, optimizer_GA, amp, version = version)
				else:
					if CONFIG["LR_STEP"]:
						save_model(ckptDir, ganet, epoch + 1, optimizer_GA, scaler, scheduler, version)
					else:
						save_model(ckptDir, ganet, epoch + 1, optimizer_GA, scaler, version=version)
			else:
				if CONFIG["LR_STEP"]:
					save_model(ckptDir, ganet, epoch + 1, optimizer_GA, lr = scheduler)
				else:
					save_model(ckptDir, ganet, epoch + 1, optimizer_GA)

	print("============================ RGANet Done Training ============================\n")


def validate(args, device):
	if CONFIG["DENORM"]: CONFIG["HAS_NORM"] = False
	# check image directory to read results
	if str(args.dir) != "results":
		if not args.dir.is_dir():
			raise IOError(f"Invalid output folder to read from:\n{args.dir.resolve()}")
	else:
		args.dir = args.dir.with_name("results" + "_acrt_" + CONFIG["DATASET"])
	# prepare dataset
	if CONFIG["DATASET"] == "suction":
		testSplitPath = args.image.parent.joinpath("test-split.txt")
		if not testSplitPath.is_file():
			raise IOError(
				f"Test-split file does not exist, please download the dataset first:\n"
				f"{trainSplitPath}")

		valData = SuctionGrasping(args.image, args.label, testSplitPath,
					  mode="test", applyTrans=False, sameTrans=False)
	else:
		raise ValueError("Unsupported dataset in this version")

	valSet = dataset.DataLoader(dataset=valData,
				 batch_size=CONFIG["TEST_BATCH"],
				 shuffle=False,
				 num_workers=CONFIG["TEST_WORKERS"],
				 pin_memory=CONFIG["TEST_PIN"],
				 drop_last=False)
	print(f"{CONFIG['DATASET']} dataset loaded.\n")
	# load network
	ganet = GANet_dense_ga_accurate_small_link(k = 15)
	if CONFIG["HAS_AMP"]: add_fp16_bn_wrapper(ganet)
	ganet.eval()
	ganet.to(device)
	# Online validate mode
	if CONFIG["ONLINE_VAL"]:
		print("\n=======================Online Validation ======================\n")
		# checkpoint filepath check
		if str(args.checkpoint) != "checkpoint":
			if not args.checkpoint.parent.is_dir():
				raise IOError(f"Designated checkpoints folder does not exist:\n"
					      f"{args.checkpoint.resolve()}")
			ckptDir = args.checkpoint.resolve()
		else:
			ckptDir = Path.cwd().joinpath("checkpoint")
			if not args.checkpoint.is_dir():
				raise IOError(f"Default folder 'checkpoint' does not exist:\n"
					      f"{args.checkpoint.resolve()}")
		fileList = sorted(ckptDir.glob("*.pt"), reverse=False,
				  key=lambda item: item.stat().st_ctime)
		totalNum = len(fileList)
		if totalNum == 0:
			raise IOError(f"Cannot find any checkpoint files in:\n"
				      f"{ckptDir.resolve()}\n")

		runMean = {"ckpt": [], "iou": [], "acc": [], "prec": [], "rec": [], "dice": []}
		evalSave = {"iou": [], "acc": [], "prec": [], "rec": [], "dice": []}
		# load all checkpoints
		for ckptIdx, ckptPath in enumerate(fileList):
			checkpoint = torch.load(ckptPath)
			ganet.load_state_dict(checkpoint['model_state_dict'])
			print(f"{ckptIdx + 1:4d}/{totalNum} checkpoint loaded ", end="")
			with torch.no_grad():
				for idx, data in enumerate(valSet):
					In = data[0].to(device)
					Out = ganet(In)
					Out = torch.softmax(Out, dim=1)
					evals = Metrics_gpu(Out.detach(), data[1].detach().to(device), one_hot=True)
					evalSave["iou"].append(evals.IOU())
					evalSave["acc"].append(evals.ACC())
					evalSave["prec"].append(evals.PRECISION())
					evalSave["rec"].append(evals.RECALL())
					evalSave["dice"].append(evals.DICE())
					print('.', end="")

				print(f" done {len(valData)} images!")
			# simple average of all batches for a checkpoint
			runMean["ckpt"].append(ckptPath.name)
			for key in evalSave.keys():
				nBatch = len(evalSave[key])
				runMean[key].append(sum(evalSave[key]) / nBatch * 100)
			# prepare for next checkpoint
			evalSave = {"iou": [], "acc": [], "prec": [], "rec": [], "dice": []}

		print("\nAll checkpoints have been evaluated.\nThis is just a guidance to select a checkpoint via batch mean, "
		      "please use offline evaluation to compute the real value with selected checkpoint.\n")
		for key in runMean.keys():
			if key != "ckpt":
				max_ = max(runMean[key])
				i = runMean[key].index(max_)
				print(f"Highest {key}: {max_:.2f}\n"
				      f"Checkpoint:{runMean['ckpt'][i]}\n"
				      f"Metircs: iou-{runMean['iou'][i]:.2f}, "
				      f"accuracy-{runMean['acc'][i]:.2f}, precision-{runMean['prec'][i]:.2f}, "
				      f"recall-{runMean['rec'][i]:.2f}, dice-{runMean['dice'][i]:.2f}\n")

		# save results if required
		if CONFIG["ONLINE_SAVE"]:
			outDir = args.dir.joinpath("evaluation", "checkpoints.txt")
			outDir.parent.mkdir(exist_ok=True, parents=True)
			if outDir.is_file():
				outDir.unlink()
				print(f"\nExisting file:\n{outDir.absolute()}\nhas been deleted\n")

			save_report(outDir, runMean, ckptDir)
			print(f"\nFull report has been saved to:\n{outDir.absolute()}\n")
	# offline testing mode, require files on disk
	else:
		if CONFIG["DATASET"] == "suction":
			print("\n================== Offline Validation ======================")
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
			# imgList = trans_img_to_cls(imgList) # for ConvCRF and FullCRF transform
			labList = trans_img_to_cls(labList)
			assert imgList and len(imgList) == len(labList), "Empty folder or length mismatch"
			L, cnt = len(imgList), 0
			# Remove original evaluation file
			evalDir = outDir.joinpath("evaluation.txt")
			if evalDir.is_file():
				evalDir.unlink()
				print(f"\nExisting file:\n{evalDir.absolute()}\nhas been deleted\n")

			for name, img in imgList.items():
				cnt += 1
				start = time.time()
				# one-hot = False for ConvCRF and FullCRF
				if CONFIG["TEST_BGR"]:
					metric = Metrics(img, labList[name], cls=0, one_hot=True)
				else:
					metric = Metrics(img, labList[name], cls=2, one_hot=True)

				metric.save_to_disk(name, outDir, interval=CONFIG["MGD_INTV"],
						    beta=CONFIG["MGD_BETA"], shift=CONFIG["MGD_CF"])
				end = time.time()
				print(f"{cnt:6d}/{L} evaluated, {(end - start) * 1e3:.3f}ms per image")

			print(f"\nEvaluation results have been saved to:\n{outDir.joinpath('evaluation.txt').resolve()}")
			print("\n===================== Done Validation =====================\n")


def main():
	args = parser.parse_args()
	# single gpu configuration
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	''' train RGANet '''
	if args.train:
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.benchmark = True
		train(args, device, VERSION)
	''' Test RGANet '''
	if args.test:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		test(args, device)
	''' Validate results '''
	if args.validate:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		validate(args, device)


if __name__ == '__main__':
	main()
