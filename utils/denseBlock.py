#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: DenseNet bottleNeck
: Author - Xi Mo
: Institute - University of Kansas
: Date -  revised on 12/24/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as ops

from utils.configuration import CONFIG

''' BottleNeck structures for different scales'''

# bottleneck
class bottleNeck(nn.Module):
    def __init__(self, inChannel, k=CONFIG["NUM_CLS"], stack=4):
        super(bottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inChannel, affine=True)
        self.bn2 = nn.BatchNorm2d(stack * k, affine=True)
        self.conv1 = nn.Conv2d(inChannel, stack * k, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(stack * k, k, 3, 1, 1, bias=True)

    def forward(self, feat):
        feat1 = ops.relu(self.bn1(feat), inplace=True)
        feat2 = self.conv1(feat1)
        feat3 = ops.relu(self.bn2(feat2), inplace=True)
        feat4 = self.conv3(feat3)
        return feat4

# bottleNeck layer for scale 1
class bottleNeck3(nn.Module):
    def __init__(self, depth=15, stack=3):
        super(bottleNeck3, self).__init__()
        self.bneck1 = bottleNeck(depth, depth, stack)
        self.bneck2 = bottleNeck(2 * depth, depth, stack)
        self.bneck3 = bottleNeck(3 * depth, depth, stack)
        self.bn = nn.BatchNorm2d(4 * depth, affine=True)

    def forward(self, feat):
        feat1 = self.bneck1(feat)
        feat2 = torch.cat((feat, feat1), dim=1)
        feat3 = self.bneck2(feat2)
        feat4 = torch.cat((feat2, feat3), dim=1)
        feat5 = self.bneck3(feat4)
        feat6 = torch.cat((feat4, feat5), dim=1)
        return self.bn(feat6)

# bottleNeck layer for scale 2
class bottleNeck6(nn.Module):
    def __init__(self, depth=15, stack=3):
        super(bottleNeck6, self).__init__()
        self.bneck1 = bottleNeck(depth, depth, stack)
        self.bneck2 = bottleNeck(2 * depth, depth, stack)
        self.bneck3 = bottleNeck(3 * depth, depth, stack)
        self.bneck4 = bottleNeck(4 * depth, depth, stack)
        self.bneck5 = bottleNeck(5 * depth, depth, stack)
        self.bneck6 = bottleNeck(6 * depth, depth, stack)
        self.bn = nn.BatchNorm2d(7 * depth, affine=True)

    def forward(self, feat):
        feat1 = self.bneck1(feat)
        feat2 = torch.cat((feat, feat1), dim=1)
        feat3 = self.bneck2(feat2)
        feat4 = torch.cat((feat2, feat3), dim=1)
        feat5 = self.bneck3(feat4)
        feat6 = torch.cat((feat4, feat5), dim=1)
        feat7 = self.bneck4(feat6)
        feat8 = torch.cat((feat6, feat7), dim=1)
        feat9 = self.bneck5(feat8)
        feat10 = torch.cat((feat8, feat9), dim=1)
        feat11 = self.bneck6(feat10)
        feat12 = torch.cat((feat10, feat11), dim=1)
        return self.bn(feat12)

# bottleNeck layer for scale 3
class bottleNeck12(nn.Module):
    def __init__(self, depth=15, stack=6):
        super(bottleNeck12, self).__init__()
        self.bneck1 = bottleNeck(depth, depth, stack)
        self.bneck2 = bottleNeck(2 * depth, depth, stack)
        self.bneck3 = bottleNeck(3 * depth, depth, stack)
        self.bneck4 = bottleNeck(4 * depth, depth, stack)
        self.bneck5 = bottleNeck(5 * depth, depth, stack)
        self.bneck6 = bottleNeck(6 * depth, depth, stack)
        self.bneck7 = bottleNeck(7 * depth, depth, stack)
        self.bneck8 = bottleNeck(8 * depth, depth, stack)
        self.bneck9 = bottleNeck(9 * depth, depth, stack)
        self.bneck10 = bottleNeck(10 * depth, depth, stack)
        self.bneck11 = bottleNeck(11 * depth, depth, stack)
        self.bneck12 = bottleNeck(12 * depth, depth, stack)
        self.bn = nn.BatchNorm2d(13 * depth, affine=True)

    def forward(self, feat):
        feat1 = self.bneck1(feat)
        feat2 = torch.cat((feat, feat1), dim=1)
        feat3 = self.bneck2(feat2)
        feat4 = torch.cat((feat2, feat3), dim=1)
        feat5 = self.bneck3(feat4)
        feat6 = torch.cat((feat4, feat5), dim=1)
        feat7 = self.bneck4(feat6)
        feat8 = torch.cat((feat6, feat7), dim=1)
        feat9 = self.bneck5(feat8)
        feat10 = torch.cat((feat8, feat9), dim=1)
        feat11 = self.bneck6(feat10)
        feat12 = torch.cat((feat10, feat11), dim=1)
        feat13 = self.bneck7(feat12)
        feat14 = torch.cat((feat12, feat13), dim=1)
        feat15 = self.bneck8(feat14)
        feat16 = torch.cat((feat14, feat15), dim=1)
        feat17 = self.bneck9(feat16)
        feat18 = torch.cat((feat16, feat17), dim=1)
        feat19 = self.bneck10(feat18)
        feat20 = torch.cat((feat18, feat19), dim=1)
        feat21 = self.bneck11(feat20)
        feat22 = torch.cat((feat20, feat21), dim=1)
        feat23 = self.bneck12(feat22)
        feat24 = torch.cat((feat22, feat23), dim=1)
        return self.bn(feat24)

# bottleNeck layer for scale 4
class bottleNeck24(nn.Module):
    def __init__(self, depth=15, stack=12):
        super(bottleNeck24, self).__init__()
        self.bneck1 = bottleNeck(depth, depth, stack)
        self.bneck2 = bottleNeck(2 * depth, depth, stack)
        self.bneck3 = bottleNeck(3 * depth, depth, stack)
        self.bneck4 = bottleNeck(4 * depth, depth, stack)
        self.bneck5 = bottleNeck(5 * depth, depth, stack)
        self.bneck6 = bottleNeck(6 * depth, depth, stack)
        self.bneck7 = bottleNeck(7 * depth, depth, stack)
        self.bneck8 = bottleNeck(8 * depth, depth, stack)
        self.bneck9 = bottleNeck(9 * depth, depth, stack)
        self.bneck10 = bottleNeck(10 * depth, depth, stack)
        self.bneck11 = bottleNeck(11 * depth, depth, stack)
        self.bneck12 = bottleNeck(12 * depth, depth, stack)
        self.bneck13 = bottleNeck(13 * depth, depth, stack)
        self.bneck14 = bottleNeck(14 * depth, depth, stack)
        self.bneck15 = bottleNeck(15 * depth, depth, stack)
        self.bneck16 = bottleNeck(16 * depth, depth, stack)
        self.bneck17 = bottleNeck(17 * depth, depth, stack)
        self.bneck18 = bottleNeck(18 * depth, depth, stack)
        self.bneck19 = bottleNeck(19 * depth, depth, stack)
        self.bneck20 = bottleNeck(20 * depth, depth, stack)
        self.bneck21 = bottleNeck(21 * depth, depth, stack)
        self.bneck22 = bottleNeck(22 * depth, depth, stack)
        self.bneck23 = bottleNeck(23 * depth, depth, stack)
        self.bneck24 = bottleNeck(24 * depth, depth, stack)
        self.bn = nn.BatchNorm2d(25 * depth, affine=True)

    def forward(self, feat):
        feat1 = self.bneck1(feat)
        feat2 = torch.cat((feat, feat1), dim=1)
        feat3 = self.bneck2(feat2)
        feat4 = torch.cat((feat2, feat3), dim=1)
        feat5 = self.bneck3(feat4)
        feat6 = torch.cat((feat4, feat5), dim=1)
        feat7 = self.bneck4(feat6)
        feat8 = torch.cat((feat6, feat7), dim=1)
        feat9 = self.bneck5(feat8)
        feat10 = torch.cat((feat8, feat9), dim=1)
        feat11 = self.bneck6(feat10)
        feat12 = torch.cat((feat10, feat11), dim=1)
        feat13 = self.bneck7(feat12)
        feat14 = torch.cat((feat12, feat13), dim=1)
        feat15 = self.bneck8(feat14)
        feat16 = torch.cat((feat14, feat15), dim=1)
        feat17 = self.bneck9(feat16)
        feat18 = torch.cat((feat16, feat17), dim=1)
        feat19 = self.bneck10(feat18)
        feat20 = torch.cat((feat18, feat19), dim=1)
        feat21 = self.bneck11(feat20)
        feat22 = torch.cat((feat20, feat21), dim=1)
        feat23 = self.bneck12(feat22)
        feat24 = torch.cat((feat22, feat23), dim=1)
        feat25 = self.bneck13(feat24)
        feat26 = torch.cat((feat24, feat25), dim=1)
        feat27 = self.bneck14(feat26)
        feat28 = torch.cat((feat26, feat27), dim=1)
        feat29 = self.bneck15(feat28)
        feat30 = torch.cat((feat28, feat29), dim=1)
        feat31 = self.bneck16(feat30)
        feat32 = torch.cat((feat30, feat31), dim=1)
        feat33 = self.bneck17(feat32)
        feat34 = torch.cat((feat32, feat33), dim=1)
        feat35 = self.bneck18(feat34)
        feat36 = torch.cat((feat34, feat35), dim=1)
        feat37 = self.bneck19(feat36)
        feat38 = torch.cat((feat36, feat37), dim=1)
        feat39 = self.bneck20(feat38)
        feat40 = torch.cat((feat38, feat39), dim=1)
        feat41 = self.bneck21(feat40)
        feat42 = torch.cat((feat40, feat41), dim=1)
        feat43 = self.bneck22(feat42)
        feat44 = torch.cat((feat42, feat43), dim=1)
        feat45 = self.bneck23(feat44)
        feat46 = torch.cat((feat44, feat45), dim=1)
        feat47 = self.bneck24(feat46)
        feat48 = torch.cat((feat46, feat47), dim=1)
        return self.bn(feat48)
