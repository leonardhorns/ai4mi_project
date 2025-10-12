#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import einsum
from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss:
    def __init__(self, smooth=1e-10):
        self.smooth = smooth

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        weak_target = weak_target.float()
        intersection = einsum("bkwh,bkwh->bk", pred_softmax, weak_target)
        denominator = einsum("bkwh->bk", pred_softmax) + einsum("bkwh->bk", weak_target)

        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)

        dice_score = dice_per_class.mean()
        loss = 1. - dice_score
        return loss


class DiceLoss2:
    def __init__(self, smooth=1e-10):
        self.smooth = smooth

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        weak_target = weak_target.float()
        intersection = einsum("bkwh,bkwh->bk", pred_softmax, weak_target)
        denominator = einsum("bkwh->bk", pred_softmax) + einsum("bkwh->bk", weak_target)

        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)

        # same as normal dice but in this case we remove empty-empty case
        valid_mask = denominator > 0
        dice_selected = dice_per_class[valid_mask]

        if dice_selected.numel() == 0:
            return torch.tensor(0.0, device=pred_softmax.device)

        dice_score = dice_selected.mean()
        loss = 1. - dice_score
        return loss


class DiceLoss3:
    def __init__(self, smooth=1e-10):
        self.smooth = smooth

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        weak_target = weak_target.float()

        # Drop background class (assumed to be channel 0)
        pred_softmax = pred_softmax[:, 1:, ...]
        weak_target = weak_target[:, 1:, ...]

        intersection = einsum("bkwh,bkwh->bk", pred_softmax, weak_target)
        denominator = einsum("bkwh->bk", pred_softmax) + einsum("bkwh->bk", weak_target)

        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_score = dice_per_class.mean()

        loss = 1. - dice_score
        return loss


class GeneralizedDiceLoss:
    def __init__(self, smooth=1e-10):
        self.smooth = smooth
        self.alpha = [1.81959727e-04, 3.77735613e-01, 2.33023099e-02, 5.09826454e-01, 8.89536640e-02]
        self.weights = torch.tensor(self.alpha, dtype=torch.float32).view(1, len(self.alpha))


    def __call__(self, pred_softmax, target_onehot):
        assert pred_softmax.shape == target_onehot.shape
        B, K, W, H = pred_softmax.shape

        p = pred_softmax.reshape(B, K, -1)
        g = target_onehot.reshape(B, K, -1)

        # class_volumes = g.sum(dim=2)  # (B, K)
        # class_volumes = torch.clamp(class_volumes, min=1.0)
        # weights = 1.0 / (class_volumes ** 2 + self.smooth)
        # weights = weights / (weights.sum(dim=1, keepdim=True) + self.smooth)

        intersection = (p * g).sum(dim=2)
        denom = p.sum(dim=2) + g.sum(dim=2)

        if self.weights.device != pred_softmax.device:
            self.weights = self.weights.to(pred_softmax.device)
        numerator = 2 * (self.weights * intersection).sum(dim=1)
        denominator = (self.weights * denom).sum(dim=1)

        dice = (numerator + self.smooth) / (denominator + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class FocalLoss:
    def __init__(self, alpha=None, gamma=2.0, eps=1e-10):
        if alpha is None:
            alpha = [0.05, 0.45, 0.05, 0.15, 0.3]

        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.eps = eps
        print(f"Initialized {self.__class__.__name__} with gamma={gamma}, alpha={alpha}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        assert simplex(pred_softmax), "Input is not a valid probability simplex"
        assert sset(weak_target, [0, 1]), "Target must be one-hot"

        B, K, W, H = pred_softmax.shape

        log_p = torch.clamp(pred_softmax, min=self.eps).log()

        # focal term: (1 - p_t)^gamma
        focal_weight = (1.0 - pred_softmax) ** self.gamma

        # apply focal weighting to log probabilities
        loss_map = -weak_target * focal_weight * log_p

        # apply per-class alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                loss_map = self.alpha * loss_map
            else:
                # alpha is a tensor of shape (K,)
                alpha = self.alpha.to(pred_softmax.device).view(1, K, 1, 1)
                loss_map = alpha * loss_map

        loss = loss_map.sum(dim=(1, 2, 3)).mean()
        return loss


# class HausdorffLoss:
#     def __init__(self, alpha=0.5, **kwargs):
#         self.loss = losses.HausdorffDTLoss()
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         loss = self.loss(pred_softmax, weak_target)
#         return loss
#
#
# class Focal2:
#     def __init__(self, alpha=None, **kwargs):
#         if alpha is None:
#             self.alpha = [1.81959727e-04, 3.77735613e-01, 2.33023099e-02, 5.09826454e-01, 8.89536640e-02]
#         else:
#             self.alpha = alpha
#
#         self.loss = losses.FocalLoss(alpha=self.alpha)
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         loss = self.loss(pred_softmax, weak_target)
#         return loss
#
#
# class GenDice2:
#     def __init__(self, **kwargs):
#         self.loss = losses.GeneralizedDiceLoss()
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         loss = self.loss(pred_softmax, weak_target)
#         return loss
#
#
# class TverskyLoss:
#     def __init__(self, **kwargs):
#         self.loss = losses.TverskyLoss()
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         loss = self.loss(pred_softmax, weak_target)
#         return loss


# class ComboLoss2:
#     def __init__(self, alpha=0.5, **kwargs):
#         print("hello!")
#         self.alpha = alpha
#
#         self.loss1 = FocalLoss(**kwargs)
#         self.loss2 = GenDice2(**kwargs)
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         task1 = self.loss1(pred_softmax, weak_target)
#         task2 = self.loss2(pred_softmax, weak_target)
#
#         return self.alpha * task1 + (1 - self.alpha) * task2
#
#
# class ComboLoss3:
#     def __init__(self, alpha=0.5, **kwargs):
#         self.alpha = alpha
#
#         self.loss1 = FocalLoss(**kwargs)
#         self.loss2 = HausdorffLoss(**kwargs)
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         task1 = self.loss1(pred_softmax, weak_target)
#         task2 = self.loss2(pred_softmax, weak_target)
#
#         return self.alpha * task1 + (1 - self.alpha) * task2
#
#
# class ComboLoss4:
#     def __init__(self, alpha=0.5, **kwargs):
#         self.alpha = alpha
#
#         self.loss1 = FocalLoss(**kwargs)
#         self.loss2 = TverskyLoss(**kwargs)
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         task1 = self.loss1(pred_softmax, weak_target)
#         task2 = self.loss2(pred_softmax, weak_target)
#
#         return self.alpha * task1 + (1 - self.alpha) * task2
#
#
# class ComboLoss5:
#     def __init__(self, alpha=None, **kwargs):
#         if alpha is None:
#             self.alpha = [0.5, 0.3, 0.2]
#         else:
#             self.alpha = alpha
#         print("hello!")
#         self.loss1 = FocalLoss(**kwargs)
#         self.loss2 = TverskyLoss(**kwargs)
#         self.loss3 = GenDice2(**kwargs)
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         task1 = self.loss1(pred_softmax, weak_target)
#         task2 = self.loss2(pred_softmax, weak_target)
#         task3 = self.loss3(pred_softmax, weak_target)
#
#         return self.alpha[0] * task1 + self.alpha[1] * task2 + self.alpha[2] * task3


class ComboLoss1:
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha

        self.loss1 = CrossEntropy(**kwargs)
        self.loss2 = DiceLoss()

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        task1 = self.loss1(pred_softmax, weak_target)
        task2 = self.loss2(pred_softmax, weak_target)

        return self.alpha * task1 + (1 - self.alpha) * task2


class ComboLoss2:
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha

        self.loss1 = CrossEntropy(**kwargs)
        self.loss2 = DiceLoss3()

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        task1 = self.loss1(pred_softmax, weak_target)
        task2 = self.loss2(pred_softmax, weak_target)

        return self.alpha * task1 + (1 - self.alpha) * task2


class ComboLoss3:
    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha

        self.loss1 = FocalLoss(**kwargs)
        self.loss2 = DiceLoss()

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape, "Shapes must match"
        task1 = self.loss1(pred_softmax, weak_target)
        task2 = self.loss2(pred_softmax, weak_target)

        return self.alpha * task1 + (1 - self.alpha) * task2


# class ComboLoss4:
#     def __init__(self, alpha=0.5, **kwargs):
#         self.alpha = alpha
#
#         self.loss1 = CrossEntropy(**kwargs)
#         self.loss2 = GeneralizedDiceLoss()
#         self.loss = losses.HausdorffDTLoss()
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         # task1 = self.loss1(pred_softmax, weak_target)
#         # task2 = self.loss2(pred_softmax, weak_target)
#         loss = self.loss(pred_softmax, weak_target)
#         return loss
#
#
# class ComboLoss5:
#     def __init__(self, alpha=0.5, **kwargs):
#         self.alpha = alpha
#
#         self.loss1 = FocalLoss(**kwargs)
#         self.loss2 = GeneralizedDiceLoss()
#
#     def __call__(self, pred_softmax, weak_target):
#         assert pred_softmax.shape == weak_target.shape, "Shapes must match"
#         task1 = self.loss1(pred_softmax, weak_target)
#         task2 = self.loss2(pred_softmax, weak_target)
#
#         return self.alpha * task1 + (1 - self.alpha) * task2
