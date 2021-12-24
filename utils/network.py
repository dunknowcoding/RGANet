#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Network frameworks and helpers
: Author - Xi Mo
: Institute - University of Kansas
: Date - 4/12/2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as ops
import time

from pathlib import Path
from utils.configuration import CONFIG
from utils.denseBlock import bottleNeck3, bottleNeck6, bottleNeck12, bottleNeck24


''' Global Attendion Module (GAM)'''

class affine_global_attention(nn.Module):
    # param 'shape' in the format [Height, Width]
    def __init__(self, shape: [int, int], C = CONFIG["NUM_CLS"], activation="sigmoid"):
        super(affine_global_attention, self).__init__()
        self.activation = activation
        H, W = shape
        kernels = [(H, 1), (W, 1), (C, 1), (1, H), (1, W), (1, C)]
        self.bn = nn.BatchNorm2d(C, affine=True)
        # C-H view
        self.conv1 = nn.Conv2d(W, W, kernels[2], groups = W, bias=True)
        self.conv2 = nn.Conv2d(W, W, kernels[3], groups = W, bias=True)
        self.bn1 = nn.BatchNorm2d(W, affine=True)
        self.bn2 = nn.BatchNorm2d(W, affine=True)
        # W-C view
        self.conv3 = nn.Conv2d(H, H, kernels[1], groups = H, bias=True)
        self.conv4 = nn.Conv2d(H, H, kernels[5], groups = H, bias=True)
        self.bn3 = nn.BatchNorm2d(H, affine = True)
        self.bn4 = nn.BatchNorm2d(H, affine = True)
        # aggregation layer
        self.aggregate = nn.Sequential(
                    nn.BatchNorm2d(2 * C, affine=False),
                    nn.Conv2d(2 * C, C, 1, 1, bias=False),
                    nn.BatchNorm2d(C, affine=False),
                    nn.SiLU(inplace=True)
                    )
        self.bn5 = nn.BatchNorm2d(C, affine=True)
        self.bn6 = nn.BatchNorm2d(C, affine=True)

    def forward(self, feat_hw):
        feat_hw = self.bn(feat_hw)
        # roll input feature map to [B, W, C, H]
        feat_ch = feat_hw.permute(0, 3, 1, 2)
        feat1 = self.bn1(self.conv1(feat_ch))
        feat2 = self.bn2(self.conv2(feat_ch))
        corr1 = torch.matmul(feat2, feat1)
        # roll back C-H correlator to [B, C, W, H]
        corr1 = corr1.permute(0, 2, 3, 1)
        corr1 = self.bn5(corr1)
        # roll C-H feature map to [B, H, W, C]
        feat_wc = feat_ch.permute(0, 3, 1, 2)
        feat3 = self.bn3(self.conv3(feat_wc))
        feat4 = self.bn4(self.conv4(feat_wc))
        corr2 = torch.matmul(feat4, feat3)
        # roll back second view to [B, C, W, H]
        corr2 = corr2.permute(0, 3, 1, 2)
        corr2 = self.bn5(corr2)
        corr3 = torch.cat((corr1, corr2), dim=1)
        if self.activation == "sigmoid":
            corr3 = torch.sigmoid(self.aggregate(corr3))
        elif self.activation == "tanh":
            corr3 = torch.tanh(self.aggregate(corr3))
        elif self.activation == "softmax":
            corr3 = torch.softmax(self.aggregate(corr3), dim=1)
        elif self.activation == "relu6":
            corr3 = ops.relu6(self.aggregate(corr3), inplace=True)
        elif self.activation == "silu":
            corr3 = ops.silu(self.aggregate(corr3), inplace=True)
        else:
            corr3 = self.aggregate(corr3)

        feat_out = torch.mul(feat_hw, corr3)
        feat_out = self.bn6(feat_out)
        return feat_out


''' RGANet-NB arch'''

class GANet_dense_ga_accurate_small_link(nn.Module):
    def __init__(self, k = 15):
        super(GANet_dense_ga_accurate_small_link, self).__init__()
        # settings
        inChannel, outChannel = 3, CONFIG["NUM_CLS"]
        H, W = CONFIG["SIZE"]
        assert not (H % 32) and not (W % 32), "SIZE setting mismatch, must be divided by 32"
        # image encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck6(k),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(7 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck12(k),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(13 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck24(k),
            nn.ReLU(inplace=True)
        )
        # image decoder
        self.upSample1 = nn.Sequential(
            nn.Conv2d(25 * k, 13 * k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.upSample2 = nn.Sequential(
            nn.Conv2d(13 * k, 7 * k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(7 * k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.upSample3 = nn.Sequential(
            nn.Conv2d(7 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.upSample4 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        # self.upSample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=False)
        self.upSample5 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.head = nn.Sequential(
            nn.Conv2d(k + inChannel, outChannel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannel, affine=True),
            nn.ReLU(inplace=True)
        )
        # upsample
        # connector
        self.connect1 = affine_global_attention([H//2, W//2], C = 4 * k, activation="sigmoid")
        self.connect2 = affine_global_attention([H//4, W//4], C = 4 * k, activation="sigmoid")
        self.connect3 = affine_global_attention([H//8, W//8], C = 7 * k, activation="sigmoid")
        self.connect4 = affine_global_attention([H//16, W//16], C = 13 * k, activation="sigmoid")
        self.connect5 = affine_global_attention([H//32, W//32], C = 25 * k, activation="sigmoid")
        self.connect6 = affine_global_attention([H, W], C=outChannel, activation="sigmoid")

        self.shrink1 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink2 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink3 = nn.Sequential(
            nn.BatchNorm2d(7 * k, affine=True),
        )
        self.shrink4 = nn.Sequential(
            nn.BatchNorm2d(13 * k, affine=True),
        )
        self.shrink5 = nn.Sequential(
            nn.BatchNorm2d(25 * k, affine=True),
        )

        self.weightSum1 = nn.Sequential(
            nn.Conv2d(13 * k * 2, 13 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum2 = nn.Sequential(
            nn.Conv2d(7 * k * 2, 7 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(7 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum3 = nn.Sequential(
            nn.Conv2d(4 * k * 2, 4 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(4 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum4 = nn.Sequential(
            nn.Conv2d(4 * k * 2, 4 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(4 * k, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        feat1 = self.conv1(img)
        feat1 = self.shrink1(torch.add(feat1, self.connect1(feat1)))
        feat2 = self.conv2(feat1)
        feat2 = self.shrink2(torch.add(feat2, self.connect2(feat2)))
        feat3 = self.conv3(feat2)
        feat3 = self.shrink3(torch.add(feat3, self.connect3(feat3)))
        feat4 = self.conv4(feat3)
        feat4 = self.shrink4(torch.add(feat4, self.connect4(feat4)))
        feat5 = self.conv5(feat4)
        feat5 = self.shrink5(torch.add(feat5, self.connect5(feat5)))
        feat6 = self.upSample1(feat5)
        feat6 = self.weightSum1(torch.cat((feat6, feat4), dim=1))
        feat7 = self.upSample2(feat6)
        feat7 = self.weightSum2(torch.cat((feat7, feat3), dim=1))
        feat8 = self.upSample3(feat7)
        feat8 = self.weightSum3(torch.cat((feat8, feat2), dim=1))
        feat9 = self.upSample4(feat8)
        feat9 = self.weightSum4(torch.cat((feat9, feat1), dim=1))
        feat10 = torch.cat((self.upSample5(feat9), img), dim=1)
        feat10 = self.connect6(self.head(feat10))
        return feat10


'''  focal loss '''

class focalLoss(nn.Module):
    def __init__(self, gamma = 2, weights=None, reduction="mean"):
        assert reduction in ["none", "mean", "sum"], "Invalid reduction option"
        super(focalLoss, self).__init__()
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.bufferFlag = False
        self.reduction = reduction
        self.scale = 10 if gamma <= 2 else 10 ** (gamma >> 1)
        if weights is not None:
            self.register_buffer("weights", torch.cuda.FloatTensor(weights))
            self.CELoss = nn.CrossEntropyLoss(weight=self.weights, reduction="none")
            self.bufferFlag = True
        else:
            self.CELoss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, output, target):
        ce_loss = self.CELoss(output, target)
        if self.bufferFlag:
            prob = torch.exp(-ops.cross_entropy(output, target, reduction="none"))
        else:
            prob = torch.exp(ce_loss)
        # for with weights or weights is None
        focal_loss = (1 - prob) ** self.gamma * ce_loss
        if self.bufferFlag: focal_loss /= self.weights.sum()
        # default: return scaled mean
        if self.reduction == "mean": return focal_loss.mean() * self.scale
        if self.reduction == "sum": return focal_loss.sum()
        return focal_loss


''' KL Divergence loss'''

class KLD(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLD, self).__init__()
        self.reduction = reduction
        self.kld = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, output, target):
        assert len(output.shape) == 4, "Must be 4-dimensional output"
        output = torch.log_softmax(output, dim=1)
        loss = self.kld(output, target)
        loss = torch.sum(loss, dim=1)
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss


'''  refer to https://github.com/NVIDIA/apex/issues/121 '''

def add_fp16_bn_wrapper(model):
    for child_name, child in model.named_children():
            classname = child.__class__.__name__
            if classname.find('BatchNorm') != -1:
                    setattr(model, child_name, BatchNormWrapper(child))
            else:
                    add_fp16_bn_wrapper(child)


''' enforce fp32 batchnorm operation, refer to https://github.com/NVIDIA/apex/issues/121 '''

class BatchNormWrapper(nn.Module):
    def __init__(self, m):
        super(BatchNormWrapper, self).__init__()
        self.m = m
        self.m.train()  # Set the batch norm to train mode

    def forward(self, x):
        inputType = x.dtype
        x = self.m(x.float())
        return x.to(inputType)


''' optimizer parser '''

def optimizer(_net):
    if CONFIG["OPTIM"] == "adamw":
        optimizer = torch.optim.AdamW(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"],
                                         amsgrad      = CONFIG["AMSGRAD"])
    elif CONFIG["OPTIM"] == "adam":
        optimizer = torch.optim.Adam(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"],
                                         amsgrad      = CONFIG["AMSGRAD"])
    elif CONFIG["OPTIM"] == "sgd":
        optimizer = torch.optim.SGD(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         momentum     = CONFIG["MOMENT"],
                                         weight_decay = CONFIG["DECAY"],
                                         dampening    = CONFIG["DAMPEN"],
                                         nesterov     = CONFIG["NESTROV"])
    elif CONFIG["OPTIM"] == "rmsprop":
        optimizer = torch.optim.RMSprop(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         momentum     = CONFIG["MOMENT"],
                                         weight_decay = CONFIG["DECAY"],
                                         alpha        = CONFIG["ALPHA"],
                                         eps          = CONFIG["EPS"],
                                         centered     = CONFIG["CENTERED"])
    elif CONFIG["OPTIM"] == "rprop":
        optimizer = torch.optim.Rprop(_net.parameters(),
                                         lr         = CONFIG["LR"],
                                         etas       = CONFIG["ETAS"],
                                         step_sizes = CONFIG["STEPSIZE"])
    elif CONFIG["OPTIM"] == "adagrad":
        optimizer = torch.optim.Adagrad(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         lr_decay     = CONFIG["LR_DECAY"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "adadelta":
        optimizer = torch.optim.Adadelta(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         rho          = CONFIG["RHO"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "sparseadam":
        optimizer = torch.optim.SparseAdam(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "adamax":
        optimizer = torch.optim.Adamax(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         betas        = CONFIG["BETAS"],
                                         weight_decay = CONFIG["DECAY"],
                                         eps          = CONFIG["EPS"])
    elif CONFIG["OPTIM"] == "asgd":
        optimizer = torch.optim.ASGD(_net.parameters(),
                                         lr           = CONFIG["LR"],
                                         lambd        = CONFIG["LAMBD"],
                                         alpha        = CONFIG["ALPHA"],
                                         weight_decay = CONFIG["DECAY"],
                                         t0           = CONFIG["T0"])
    else:
        raise NameError(f"Unsupported optimizer {CONFIG['OPTIM']}, please customize it.")

    return optimizer


''' Helper to comput output padding size '''

def _get_deconv_out_padding(inSize: list, outSize: list, kernel: tuple, stride: tuple,
                            padding=(0, 0), dilation=(1, 1)) -> tuple:
    assert inSize[0] <= outSize[0] and inSize[1] <= outSize[1], "Output size must at least equal to input size"
    assert len(inSize) == 2 and len(outSize) == 2, "In-out sizes must be 2-dimension tuple or list"
    padH = outSize[0] - 1 - (inSize[0] - 1) * stride[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1)
    padW = outSize[1] - 1 - (inSize[1] - 1) * stride[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1)
    return tuple((padH, padW))


''' Write model to disk '''
def save_model(baseDir: Path, network: torch.nn.Module, epoch: int,
               optimizer: torch.optim, amp=None, lr=None, version=1.9, postfix=CONFIG['MODEL']):
    if CONFIG["HAS_AMP"]:
        if version < 1.6:
            date = time.strftime(f"{CONFIG['DATASET']}-%Y%m%d-%H%M%S-Epoch-{epoch}_{postfix}_"
                                 f"{'O' + str(CONFIG['AMP_LV'])}.pt", time.localtime())
            path = baseDir.joinpath(date)
            print("\nNow saveing model to:\n%s" % path)
            if CONFIG["LR_STEP"]:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'scheduler_state_dict': lr.state_dict()
                    }, path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'amp': amp.state_dict()
                    }, path)
        else:
            date = time.strftime(f"{CONFIG['DATASET']}-%Y%m%d-%H%M%S-Epoch-{epoch}_{postfix}_AMP.pt", time.localtime())
            path = baseDir.joinpath(date)
            print("\nNow saveing model to:\n%s" % path)
            if CONFIG["LR_STEP"]:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': amp.state_dict(),
                    'scheduler_state_dict': lr.state_dict()
                    }, path)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': amp.state_dict()
                    }, path)
    else:
        date = time.strftime(f"{CONFIG['DATASET']}-%Y%m%d-%H%M%S-Epoch-{epoch}_{postfix}.pt", time.localtime())
        path = baseDir.joinpath(date)
        print("\nNow saveing model to:\n%s" % path)
        if CONFIG["LR_STEP"]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr.state_dict()
            }, path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
    print("Done!\n")
