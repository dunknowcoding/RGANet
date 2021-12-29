#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Configuraion profile
: Author - Xi Mo
: Institute - University of Kansas
: Date - revised on 12/24/2021
"""

import argparse
import math
from pathlib import Path


# Training
parser = argparse.ArgumentParser("RGANet Parser")
parser.add_argument("-train", action = "store_true",
                    help = "Train network only")
parser.add_argument("-r", "--restore", action = "store_true",
                    help = "Restore training by loading specified checkpoint or lattest checkpoint")

# Accomodation to suction dataset
parser.add_argument("-i", "--image", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/color-input",
                    help = "Directory to training images")
parser.add_argument("-l", "--label", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/label",
                    help = "Directory to training annotations")
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by users")
parser.add_argument("-d", "--dir", type = Path, default = r"results",
                    help = r"Valid for train and test GCRF mode - specify the folder to read/save, or "
                           r"valid for test mode, specify the folder to save results and labels, or "
                           r"valid for validate mode, specify the source to load images/save results")

# Testing and validation
parser.add_argument("-test", action = "store_true",
                    help = "Test and visualize only")
parser.add_argument("-v", "--validate", action = "store_true",
                    help = "Validate tesing results using metrics")

CONFIG = {
    "DATASET":  "suction",               # choose dataset, we only provide this option
                                                          
    "POSTFIX": ".png",                       # label/sample image postfix to read or save as
    "SIZE": (480, 640),                         # input size specification: (H, W), must be divisible by 32
    "HAS_NORM": False,                                                         # normailzationm,for samples only
    "PAR_NORM": {"mean": (0.485, 0.456, 0.406),             # dictionary format with tuples
                 "std": (0.229, 0.224, 0.225)},                               # valid for train and test

    # RGANet Training
    # Loss function
    "LR_STEP": None,            # list to specify intervals for weights decay,  None to turn off multi-step learning rate
    "LR_MUL": 0.5,                 # set multiplicative factor of learning rate decay if LR_STEP is not None
    "LOSS": "ce",                    # choose loss function between "focal", "ce", "bce", "huber",
                                                # "poisson", "kld" (same as cross-entropy using one-hot encoding labels)
    # "WEIGHT": [0.063, 0.266, 0.671],      # adaptive weights for ce and focal loss
    "WEIGHT": [0.25, 0.25, 0.5],                  # lucky weights for ce and focal loss

    # bce losses
    "GAMMA": 1.5,                 # gamma for focal loss
    "BETA": 0.618,                   # beta for huber loss
    "FULL": True,                     # add the Stirling approximation term to Poisson loss
    "PEPS": 1e-8,                     # eps for Poisson loss
    "REDUCT": "mean",          # loss reduction method, "mean", "sum" , "none"

    # optimizer
    "OPTIM": "adamw",                 # "sgd", "adam", "adamw", "rmsprop", "rprop", "adagrad", "adadelta"
                                                        # and "sparseadam", "adamax", "asgd"
    "LR": 0.00015,                          # global learning rate/initial learning rate
    "BETAS": (0.9, 0.999),              # coefficients for computing running averages of gradient and its square
    "EPS": 1e-08,                            # term added to the denominator to improve numerical stability
    "DECAY": 0,                               # weight decay (L2 penalty)
    "AMSGRAD": True,                 # use the AMSGrad variant
    "MOMENT": 0,                         # momentum factor
    "DAMPEN": 0,                          # dampening for momentum
    "NESTROV": False,                  # enables Nesterov momentum
    "ALPHA": 0.99,                        # smoothing constant
    "CENTERED": False,                # gradient is normalized by estimation of variance
    "ETAS": (0.5, 1.2),                   # multiplicative increase and decrease factors (etaminus, etaplis)
    "STEPSIZE": (1e-06, 50),        # minimal and maximal allowed step sizes
    "LR_DECAY": 0,                       # learning rate decay
    "RHO": 0.9,                              # coefficient for computing a running average of squared gradients
    "LAMBD": 1e-4,                      # decay term
    "T0": 1e6,                                 # point at which to start averaging

    # training
    "IGNORE": True,                       # True to ignore the class NUM_CLS-1, for cityscape and camvid datasets
    "VAL_BATCH": False,                # turn on/off online valiation of training batch
    "VAL_CLS": 3,                            # trainId of cityscape dataset to evaluate during training, 18 - bicycle
                                                         # for suction dataset, only category 2 is evaluated, and this cfg is ignored
    "HAS_AMP": True,                    # True to turn on automatic mixed precision, otherwise using default FP32 precision
    "AMP_LV": 1,                             # set amp levels, valid for pytorch < 1.6 only, valued from [0, 1, 2, 3], 1 is recommended
    "AUGMENT": True,                   # enforce random augmentation upon batches
    "BATCHSIZE": 24,                      # batchsize for training
    "EPOCHS": 3000,                       # epoches for training
    "SHOW_LOSS": 10,                   # number of minibatchs processed to print training info
    "SAVE_MODEL": 2,                    # epoch intervel to save, start counting from epoch 1
    "SHUFFLE": True,                       # random shuffle
    "NUM_WORKERS": 0,               # set to 0 if memorry error occurs
    "PIN_MEMORY": True,             # set to false if memory is insufficient
    "DROP_LAST": False,
    "NUM_CLS": 3,                         # valid number of classes for RGANet use only
    "INT_CLS": (255, 0, 128),        # raw label intensity levels for RGANet saved results (suction)

    # RGANet Testing
    "DENORM": False,                  # set True to disable normalization
    "TEST_BATCH": 100,              # batchsize for testing
    "TEST_RUNTIME": False,       # False to disable runtime test
    "TEST_MUL": 5,                      # set a multiplier for testing
    "TEST_TIME": 1,                      # show runtime stats every specified number of testing batches
    "TEST_WORKERS": 0,            # set number of workers to run testing batches
    "TEST_PIN": True,                  # set to True if memory is pinned for testing batches
    "TEST_SAVE": True,               # if tests are done, save results and their labels to disk
    "TEST_BGR": True,                 # Test: True - target class save as bright color, otherwise cold color
                                                      # Validation: True - evaluate pred class 0, otherwise NUM_CLS-1
                                                      # True for suction dataset

    # RGANet Validation.
    # MGRID metric for "suction" dataset
    "MGD_INTV": (12, 12),         # set intervals (dH, dW) for metric MGRID
    "MGD_BETA": 0.5,                # beta for metric MGRID
    "MGD_CF": (0.525, 0.5),      # (Cm, Fm) for metric MGRID
                                        # 0 for RGANet
    "LAB_CLS": 0,            # 2 for ChenXY's experiments, interested label class to evaluate, range from 0 to NUM_CLS-1
                                        # MUST pair with correct value in label, for 'suction' dataset and offline mode only
    "ONLINE_VAL": False,  # Set True to use online evaluation for all checkpoints,
                                            # only show the best checkpoints, set False to use offline mode
                                            # Test configurations are effective for online validation except
                                            # "TEST_TIME", "TEST_SAVE" and "TEST_RUNTIME", "TEST_BGR"(will
                                            # always evaluate class "NUM_CLS"-1 for both pred and gt)
    "ONLINE_SAVE": True,    # True to save full report as "checkpoints.txt" to arg '-d'
                                                # specified folder, will be turned off if ONLINE_VAL is False

    # Augmentation
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
