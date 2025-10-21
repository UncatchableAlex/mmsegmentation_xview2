# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              class_weights: Union[torch.Tensor, None] = None,
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        class_weights (torch.Tensor or list, optional): Per-class weights
            used to average class-wise dice. Length should be C (before
            removing ignore_index). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.
    """
     # expect pred and target to have shape (N, C, H, W)
    num_classes = pred.shape[1]
    if ignore_index is not None:
        mask = torch.arange(num_classes, device=pred.device) != ignore_index
        pred = pred[:, mask, :, :]
        target = target[:, mask, :, :]
        # adjust class_weights if provided
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, device=pred.device, dtype=pred.dtype)
            if cw.numel() == num_classes:
                cw = cw[mask]
            else:
                # if provided weights length doesn't match original classes,
                # try to use as-is (will be validated later)
                cw = cw
            class_weights = cw
        assert pred.shape[1] != 0  # if the ignored index is the only class
    
    # reshape to (N, C, S) where S = H*W
    input = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target = target.reshape(target.shape[0], target.shape[1], -1).float()

    # input = pred.flatten(1)
    # target = target.flatten(1).float()

    # per-sample per-class intersection / sums
    a = torch.sum(input * target, dim=2)  # (N, C)
    if naive_dice:
        b = torch.sum(input, dim=2)
        c = torch.sum(target, dim=2)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, dim=2) + eps
        c = torch.sum(target * target, dim=2) + eps
        d = (2 * a) / (b + c)

    loss_per_class = 1 - d  # (N, C)

    # handle class weighting when averaging across classes
    if class_weights is not None:
        cw = torch.as_tensor(class_weights, device=loss_per_class.device, dtype=loss_per_class.dtype)
        # check shape of cw
        if cw.ndim == 1 and cw.numel() == loss_per_class.shape[1]:
            # check sum of cw
            if cw.sum() != 1:
                raise ValueError("class_weights dont sum to 1")
            loss = (loss_per_class * cw.unsqueeze(0)).sum(dim=1)  # (N,)
        else:
            raise ValueError("class_weights must be a 1D tensor/list with length equal to number of classes")
    else:
        loss = loss_per_class.mean(dim=1)  # average over classes -> (N,)

    # sample-wise weight & reduction
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 class_weights=None,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.class_weights = class_weights
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            weight,
            class_weights=self.class_weights,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)
        
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
