import torch
import torch.nn as nn

class dice_bce_loss(nn.Module):
    """
    Dice Coefficient binary cross entropy loss function.

    This class defines the combined Dice Coefficient and binary cross-entropy loss function.

    Args:
        batch (bool): Whether to compute the loss for a batch of data.

    Returns:
        torch.Tensor: Computed loss value.
    """

    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a


def acc_sen(pred, mask):
    """
    Calculate accuracy and sensitivity metrics.

    This function computes accuracy and sensitivity metrics based on predicted and actual masks.

    Args:
        pred (torch.Tensor): Predicted mask.
        mask (torch.Tensor): Actual mask.

    Returns:
        tuple: Tuple containing calculated accuracy, sensitivity, precision, F1 score, and IoU.
    """

    pred = torch.round(pred)
    TP = (mask * pred).sum(1).sum(1).sum(1)
    TN = ((1 - mask) * (1 - pred)).sum(1).sum(1).sum(1)
    FP = pred.sum(1).sum(1).sum(1) - TP
    FN = mask.sum(1).sum(1).sum(1) - TP
    acc = (TP + TN) / (TP + TN + FP + FN)
    acc = torch.sum(acc)

    sen = TP / (TP + FN)
    sen = torch.sum(sen)

    prec = TP / (TP + FP)
    prec = torch.sum(prec)

    f1 = 2 * TP / (2 * TP + FP + FN)
    f1 = torch.sum(f1)

    iou = TP / (TP + FP + FN)
    iou = torch.sum(iou)

    return acc, sen, prec, f1, iou
