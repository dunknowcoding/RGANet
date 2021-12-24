#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - Comparision experiments
: Splite evaluation texts into individual text files for SPSS
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/13/2021 last updated 6/13/2021
"""

from pathlib import Path

evalPath = Path(r"E:\RGANet\results_acrt\evaluation\evaluation.txt").resolve()
splitPath = Path(r"E:\RGANet\results_acrt\evaluation\metircs").resolve()

splitPath.mkdir(exist_ok=True, parents=True)
splitPath.joinpath("acc.txt").unlink(missing_ok=True)
splitPath.joinpath("dice.txt").unlink(missing_ok=True)
splitPath.joinpath("fpr.txt").unlink(missing_ok=True)
splitPath.joinpath("iou.txt").unlink(missing_ok=True)
splitPath.joinpath("mcc.txt").unlink(missing_ok=True)
splitPath.joinpath("mgrid.txt").unlink(missing_ok=True)
splitPath.joinpath("name.txt").unlink(missing_ok=True)
splitPath.joinpath("precision.txt").unlink(missing_ok=True)
splitPath.joinpath("recall.txt").unlink(missing_ok=True)
splitPath.joinpath("tpr.txt").unlink(missing_ok=True)

if __name__ == '__main__':
    with open(evalPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            oneItem = line.split(sep=" ")
            for idx, item in enumerate(oneItem):
                if idx == 0:
                    t = ["name", item]
                else:
                    t = item.split(sep=":")

                fileName = t[0] + ".txt"
                filePath = splitPath.joinpath(fileName)
                with open(filePath, 'a+') as g:
                    g.write(f"{t[1]}\n")

    print("Done spliting!")
