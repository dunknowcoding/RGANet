"""
: Project - Comparision experiments
: Convert image to BGR mode and show averages of metircs
: Author - Xi Mo
: Institute - University of Kansas
: Date - 6/16/2021
"""

from PIL import Image
from pathlib import Path
from utils.validate import Metrics
from utils.configuration import CONFIG
import numpy as np

# Specify directories to the anotations and predictions to be evaluated
gtDir = Path(r"D:\RGANet\results_dense_backup\annotations").resolve()
pdDir = Path(r"D:\RGANet\experiments\stdc\outputs_stdc1").resolve()

INT_LV = (3, 0, 1)    # set intensity levels in predications, (bg, ns, eval_cls)
SAVE_IMG = True       # set True to save transformed images
saveDir = Path(r"D:\RGANet\experiments\stdc\evaluation").resolve()

def get_max_images(path: Path, save: Path):
    test = Path(r"D:\RGANet\results_acrt_with_aug_NB_small_link\output")
    save = Path(r"D:\RGANet\results_acrt_with_aug_NB_small_link\output_max")
    get_max_images(test, save)
    files = path.glob("*.png")
    for f in files:
        im = np.array(Image.open(f), 'f')
        H, W, _ = im.shape
        im = im.argmax(axis=-1)
        im = np.expand_dims(im, axis= -1)
        dummy = np.repeat(im, 3, axis=-1)
        for h in range(H):
            for w in range(W):
                if dummy[h, w, 0] == 0:
                    dummy[h,w, :] = (255, 0, 0)
                elif dummy[h, w, 0] == 2:
                    dummy[h, w, :] = (0, 0, 255)
                elif dummy[h, w, 0] == 1:
                    dummy[h, w, :] = (0, 255, 0)

        dummy = Image.fromarray(np.uint8(dummy))
        filename = save.joinpath(f.name)
        dummy.save(filename)


if __name__ == '__main__':
    # overwrite the configuration parameters
    CONFIG["NUM_CLS"] = 3
    CONFIG["LAB_CLS"] = 2
    CONFIG["SIZE"] = (480, 640)

    gtList, pdList = Path.glob(gtDir, "*.png"), Path.glob(pdDir, "*.png")
    img, lab = dict(), dict()
    stats = {"iou": [], "acc": [], "prec": [], "rec": [], "dice": [], "mcc": [], "mgrid": []}
    print("\n Processing images, pLease wait ... \n")
    for p in gtList:
        im = np.array(Image.open(p).convert('L'), 'f') * CONFIG["NUM_CLS"] / 256
        lab[p.stem] = im.astype('uint8')

    for idx, p in enumerate(pdList):
        im = np.array(Image.open(p).convert('L'), 'f')
        t = im.max()
        im = im / t * CONFIG["NUM_CLS"]
        img[str(idx+1)] = im.astype('uint8') # Chen's
        # img[p.stem] = im.astype('uint8') # BiSeNet

    for name, label in lab.items():
        metric = Metrics(img[name], label, INT_LV[2])
        stats["iou"].append(metric.IOU())
        stats["acc"].append(metric.ACC())
        stats["prec"].append(metric.PRECISION())
        stats["rec"].append(metric.RECALL())
        stats["dice"].append(metric.DICE())
        stats["mcc"].append(metric.MCC())
        stats["mgrid"].append(metric.MGRID(interval = CONFIG["MGD_INTV"],
                              beta = CONFIG["MGD_BETA"], shift = CONFIG["MGD_CF"]))

    for k in stats.keys(): stats[k] = np.array(stats[k])
    print(f"\nStatistics:\n"
          f"Jaccard: {stats['iou'].mean() * 100:.2f}\n"
          f"Accuracy: {stats['acc'].mean() * 100:.2f}\n"
          f"Precision: {stats['prec'].mean() * 100:.2f}\n"
          f"Recall: {stats['rec'].mean() * 100:.2f}\n"
          f"Dice: {stats['dice'].mean() * 100:.2f}\n"
          f"mcc: {stats['mcc'].mean() * 100:.2f}\n"
          f"MGRID: {stats['mgrid'].mean() * 100:.2f}\n")

    if SAVE_IMG:
        print("\nNow saving transformed images to disk.")
        saveDir.mkdir(exist_ok=True, parents=True)
        for name, im in img.items():
            dummy = np.zeros((im.shape[0], im.shape[1], 3))
            bgCoords = np.where(im == INT_LV[0])
            dummy[bgCoords[0], bgCoords[1], 2] = 255
            nsCoords = np.where(im == INT_LV[1])
            dummy[nsCoords[0], nsCoords[1], 1] = 255
            ecCoords = np.where(im == INT_LV[2])
            dummy[ecCoords[0], ecCoords[1], 0] = 255
            dummy = dummy.astype('uint8')
            Image.fromarray(dummy).save(saveDir.joinpath(name + ".png"))

        print("Done!")
