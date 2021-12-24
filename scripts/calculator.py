#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: calculate statistics
: Author - Xi Mo
: Institute - University of Kansas
: Date - 5/20/2021
: HowTo: Set "Choice", evaluation file is a must-be for calculating statistics
"""
import torch
from pathlib import Path
import numpy as np
from thop import profile


Choice = {

    "COUNT_PARAMS": 0,      # Set True to see comparision on parameters
    "CAL_STATS": 1,         # Show statistics, NOTE: "evaluation.txt" is necessary
    "EVA_DIR": r"D:\RGANet\results_acrt_suction\evaluation\evaluation.txt",
                            # path to "evalutation.txt"
    "PERCENTI": np.linspace(5, 100, 19, dtype=int, endpoint=False),
                            # set percentile, 0-100, can either be a number
    "SAVE_STATS": 0,        # set True to save statistic results to the same folder
                            # ignored by setting CLA_STATS to 0
    "SHOW_MACs": 0,         # set true to show GFLOPs info of model using thop
    "MODEL": "acrt"        # choose model to calculate GFLOPs between "acrt" and "fast"
}

# get network parameters
def cal_param(_net):
    return {
        "total": sum(item.numel() for item in _net.parameters()) / 1e6,
        "train": sum(item.numel() for item in _net.parameters()
                     if item.requires_grad) / 1e6
        }


if __name__ == '__main__':
    # calculate parameters
    if Choice["COUNT_PARAMS"]:
        # from utils.network import GANet_sep, GANet_shared, GANet_shared_ga_realtime, \
        #     GANet_shared_ga_accurate, GANet_dense_ga_accurate
        from utils.network import GANet_dense_ga_accurate_small_link, GANet_dense_ga_realtime
        from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, \
                  deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

        modelLList = torch.nn.ModuleDict({
                    "RGA_accurate": GANet_dense_ga_accurate_small_link(20),
                    "RGA_realtime": GANet_dense_ga_realtime(),
                    "FCN_resnet50": fcn_resnet50(),
                    "FCN_resnet101": fcn_resnet101(),
                    "DeepLabv3_resnet50": deeplabv3_resnet50(),
                    "DeepLabv3_resnet101": deeplabv3_resnet101(),
                    "DeepLabv3_mobilenetv3_large": deeplabv3_mobilenet_v3_large(),
                    "LR-ASPP_mobilenet_v3_large": lraspp_mobilenet_v3_large()
                })
        print("\n============== RGANet Statistics: Parameters ==============\n")
        for key in modelLList.keys():
            stats = cal_param(modelLList[key])
            print(f"Model: {key}\nTotal: {stats['total']: .2f}M\n"
                  f"Trainable: {stats['train']: .2f}M\n")

    # stats of evaluation performance for all metrics
    if Choice["CAL_STATS"]:
        textDir = Path.resolve(Path(Choice["EVA_DIR"]))
        assert textDir.is_file(), "Cannot find evaluation file"
        print("\n============== RGANet Statistics: evaluation ==============")
        with open(textDir, "rt") as f:
            Lines = f.read().splitlines()

        assert len(Lines) > 0, "Empty evaluation file"
        varDict = {"name": [], "iou":[], "acc": [], "dice":[], "precision": [],
                   "recall":[], "mgrid": [], "tpr": [], "fpr":[], "mcc": []}
        for l in Lines:
            items = l.split(" ")
            varDict["name"].append(items[0])
            for item in items[1:]:
                varDict[item.split(":")[0]].append(float(item.split(":")[1]))

        All = dict()
        for key in varDict.keys():
            if key != "name" and len(varDict[key]):
                All[key] = np.array(varDict[key])

        All["STEP"] = Choice["PERCENTI"]

        for key in All.keys():
            if key != "STEP":
                All[key] = {
                        "range": All[key].ptp(),
                        "percenti": np.percentile(All[key], All["STEP"]),
                        "mean": All[key].mean(),
                        "median": np.median(All[key]),
                        "var": All[key].var(),
                        "std": All[key].std()
                        }

                print(f"\n***** {key} *****\n"
                      f"Range: {All[key]['range']:.2f}\n"
                      f"Mean: {All[key]['mean']:.2f}\n"
                      f"Std: {All[key]['std']:.2f}\n"
                      f"Median: {All[key]['median']:.2f}\n"
                      f"Var: {All[key]['var']:.2f}\n"
                      f"Step:\n{All['STEP']}\n"
                      f"Percentile:\n{All[key]['percenti']}\n")

        # save results
        if Choice["SAVE_STATS"]:
            statsDir = textDir.parent.joinpath("statistics.txt")
            print(f"\nSaving statistic results to:\n{statsDir}\n")
            with open(statsDir, "a+") as f:
                f.write(f"\n\n\n\nevaluation file:\n{textDir.resolve()}\n")
                f.write(f"\nPercentile percentages:\n{All['STEP']}\n")
                for key in All.keys():
                    if key != "STEP":
                        f.write(f"\n***** {key} *****\n"
                                f"Range: {All[key]['range']:.2f}\n"
                                f"Mean: {All[key]['mean']:.2f}\n"
                                f"Std: {All[key]['std']:.2f}\n"
                                f"Median: {All[key]['median']:.2f}\n"
                                f"Var: {All[key]['var']:.2f}\n"
                                f"Percentile:\n{All[key]['percenti']}\n")

            print(f"Done!\n")
    # show MACs info
    if Choice["SHOW_MACs"]:
        assert Choice["MODEL"] in ["acrt", "fast"]
        if Choice["MODEL"] == "acrt":
            from utils.network import GANet_dense_ga_accurate_small_link
            model = GANet_dense_ga_accurate_small_link(20)
        elif Choice["MODEL"] == "fast":
            from utils.network import GANet_dense_ga_realtime
            model = GANet_dense_ga_realtime

        input_ = torch.randn(1, 3, 512, 1024)
        macs = profile(model, inputs=(input_,))
        print(f"\n{'RGANet_'+Choice['MODEL']}:\n"
              f"{macs[0]/5e8:.4f}G\n"
              f"{macs[1]/1e6:.4f}M parameters\n")



