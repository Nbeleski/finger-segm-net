import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Loss functions
# ------------------------
def dice_loss(y_pred, y_true, eps=1e-8):
    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def seg_loss(y_pred, y_true, lamb_smooth=0.5, lamb_dice=1.0):
    eps = 1e-8
    # FIXME:
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    total_elements = y_true.numel()
    label_pos = (y_true > 0).float()
    lamb_pos = 0.5 * total_elements / (label_pos.sum() + eps)
    lamb_neg = 1. / (2. - 1. / (lamb_pos + eps))

    logloss = lamb_pos * y_true * torch.log(y_pred) + lamb_neg * (1 - y_true) * torch.log(1 - y_pred)
    logloss = -logloss.sum() / total_elements

    dice = dice_loss(y_pred, y_true)

    smooth_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device=y_pred.device).view(1, 1, 3, 3) / 8
    smoothness = F.conv2d(y_pred, smooth_kernel, padding=1)
    smooth_loss = torch.mean(torch.abs(smoothness))

    return logloss + lamb_dice * dice + lamb_smooth * smooth_loss

def bce_loss(pred, target, mask=None, eps=1e-8):
    pred = torch.clamp(pred, eps, 1 - eps)
    # Make loss positive by negating log-prob expression up front
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + eps)
    return loss.mean()

# TODO: try different params ex: 0.5, 1.5
def focal_loss(pred, target, mask=None, alpha=0.25, gamma=2.0, eps=1e-8):
    pred = torch.clamp(pred, eps, 1 - eps)
    pt = pred * target + (1 - pred) * (1 - target)
    pt = torch.clamp(pt, eps, 1.0 - eps)  # <-- important fix
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + eps)
    return loss.mean()

# def compute_multitask_loss(pred, target):
#     losses = {}
#     losses['seg_loss']  = seg_loss(pred['segmentation'], target['seg_out'])

#     valid_mask = (target['mnt_s_out'] >= 0)  # include both 0 and 1, exclude -1

#     # Only compute BCE where target is not -1 (ignore regions)
#     losses['mnt_s_loss'] = focal_loss(pred['minutiae']['mnt_s_score'], target['mnt_s_out'], mask=valid_mask)
#     mask = (target['mnt_s_out'] > 0)  # only for orientation/offsets (positives only)
#     losses['mnt_o_loss'] = bce_loss(pred['minutiae']['mnt_o_score'], target['mnt_o_out'], mask)
#     losses['mnt_w_loss'] = bce_loss(pred['minutiae']['mnt_w_score'], target['mnt_w_out'], mask)
#     losses['mnt_h_loss'] = bce_loss(pred['minutiae']['mnt_h_score'], target['mnt_h_out'], mask)
#     return losses

def compute_multitask_loss(pred, target):
    losses = {}
    losses['seg_loss']  = seg_loss(pred['segmentation'], target['seg_out'])

    # 1. Valid regions (exclude -1s)
    valid_mask = (target['mnt_s_out'] >= 0).float()  # shape (B, 1, 64, 64)

    # 2. Segmentation-based soft weighting
    seg_weight = pred['segmentation'].detach()       # shape (B, 1, 64, 64)
    seg_weight = F.interpolate(pred['segmentation'].detach(), size=target['mnt_s_out'].shape[-2:], mode='bilinear', align_corners=False)

    # 3. Combined mask = valid regions * ridge confidence
    mask_score = valid_mask.float() * seg_weight

    losses['mnt_s_loss'] = focal_loss(pred['minutiae']['mnt_s_score'], target['mnt_s_out'], mask=mask_score)

    # 4. For orientation and offset, use only positive samples (where target == 1)
    mask_pos = (target['mnt_s_out'] > 0).float()
    losses['mnt_o_loss'] = bce_loss(pred['minutiae']['mnt_o_score'], target['mnt_o_out'], mask_pos)
    losses['mnt_w_loss'] = bce_loss(pred['minutiae']['mnt_w_score'], target['mnt_w_out'], mask_pos)
    losses['mnt_h_loss'] = bce_loss(pred['minutiae']['mnt_h_score'], target['mnt_h_out'], mask_pos)

    return losses

def total_loss_from_dict(losses):
    return (
        5.0  * losses["seg_loss"] + # was 10.
        500.0 * losses["mnt_s_loss"] + # was 200.
        0.5   * losses["mnt_o_loss"] +
        0.5   * losses["mnt_w_loss"] +
        0.5   * losses["mnt_h_loss"]
    )
