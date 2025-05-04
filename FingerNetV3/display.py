import numpy as np
import cv2
from scipy import ndimage

# -------------------------------------------------------------
# Minutiae post-processing
# -------------------------------------------------------------
# def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5, cell_size=8):
#     mnt_s_out = np.squeeze(mnt_s_out)
#     mnt_w_out = np.squeeze(mnt_w_out)
#     mnt_h_out = np.squeeze(mnt_h_out)
#     mnt_o_out = np.squeeze(mnt_o_out)
#     assert mnt_s_out.ndim == 2
#     H, W = mnt_s_out.shape
#     n_angle = mnt_o_out.shape[0]
#     bin_width = 2 * np.pi / n_angle
#     mask = mnt_s_out > thresh
#     rows, cols = np.where(mask)
#     if len(rows) == 0:
#         return np.zeros((0, 4))
#     mnt_w_idx = np.argmax(mnt_w_out[:, rows, cols], axis=0)
#     mnt_h_idx = np.argmax(mnt_h_out[:, rows, cols], axis=0)
#     mnt_o_idx = np.argmax(mnt_o_out[:, rows, cols], axis=0)
#     angles_rad = mnt_o_idx * bin_width  # now [0, 2π)
#     xs = cols * cell_size + mnt_w_idx
#     ys = rows * cell_size + mnt_h_idx
#     scores = mnt_s_out[rows, cols]

#     minutiae = np.stack([xs, ys, angles_rad, scores], axis=1)
#     return minutiae

def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5, cell_size=8, quality_map=None):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert mnt_s_out.ndim == 2
    H, W = mnt_s_out.shape
    n_angle = mnt_o_out.shape[0]
    bin_width = 2 * np.pi / n_angle

    if quality_map is not None:
        # TODO: sqrt the viz
        quality_map = np.clip(np.squeeze(quality_map), 0.0, 1.0)
        quality_weight = np.sqrt(quality_map)  # or quality_map ** 0.5
        mnt_s_out = mnt_s_out * quality_weight
        # mnt_s_out = mnt_s_out * np.squeeze(quality_map)

    mask = mnt_s_out > thresh
    # print(mask.shape, quality_map.shape, mnt_s_out.shape)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((0, 4))

    mnt_w_idx = np.argmax(mnt_w_out[:, rows, cols], axis=0)
    mnt_h_idx = np.argmax(mnt_h_out[:, rows, cols], axis=0)
    mnt_o_idx = np.argmax(mnt_o_out[:, rows, cols], axis=0)
    angles_rad = mnt_o_idx * bin_width
    xs = cols * cell_size + mnt_w_idx
    ys = rows * cell_size + mnt_h_idx
    scores = mnt_s_out[rows, cols]
    minutiae = np.stack([xs, ys, angles_rad, scores], axis=1)
    return minutiae

def nms_minutiae(minutiae, dist_thresh=15):
    """
    Apply non-maximum suppression to decoded minutiae.
    Arguments:
        minutiae: (N,4) array [x, y, theta, score]
        dist_thresh: distance threshold in pixels
    Returns:
        filtered_minutiae: (M,4) array
    """
    if len(minutiae) == 0:
        return minutiae

    # Sort minutiae by score descending
    idxs = np.argsort(-minutiae[:, 3])
    minutiae = minutiae[idxs]

    selected = []
    used = np.zeros(len(minutiae), dtype=bool)

    for i in range(len(minutiae)):
        if used[i]:
            continue
        selected.append(minutiae[i])
        # Suppress minutiae too close to this one
        dists = np.sqrt(np.sum((minutiae[:, :2] - minutiae[i, :2])**2, axis=1))
        used[dists < dist_thresh] = True
        used[i] = False  # keep itself

    return np.array(selected)

#
# Quality
#
def overlay_quality_map(base_img, quality_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay a [0,1] quality map on a grayscale base image."""
    # Ensure quality_map is (H, W) and float
    if quality_map.ndim == 3 and quality_map.shape[0] == 1:
        quality_map = quality_map[0]
    elif quality_map.ndim == 3 and quality_map.shape[2] == 1:
        quality_map = quality_map[:, :, 0]

    quality_map = np.clip(quality_map, 0.0, 1.0)
    quality_map_uint8 = (quality_map * 255).astype(np.uint8)  # shape (H, W)

    # Now apply colormap safely
    heatmap = cv2.applyColorMap(quality_map_uint8, colormap)

    # Make sure base image is also uint8 and 3-channel
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    elif base_img.shape[2] == 1:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(base_img, 1 - alpha, heatmap, alpha, 0)



# -------------------------------------------------------------
# Drawing functions
# -------------------------------------------------------------
def draw_segmentation_comparison(seg_label, seg_pred, thr=0.5):
    seg_label_bin = (seg_label > thr).astype(np.uint8) * 255
    seg_pred_bin = (seg_pred > thr).astype(np.uint8) * 255
    seg_label_img = cv2.resize(seg_label_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
    seg_pred_img = cv2.resize(seg_pred_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
    concat = np.hstack([seg_label_img, seg_pred_img])
    return concat.astype(np.uint8)

def draw_minutiae_on_image(image, minutiae, color=(0, 0, 255)):
    img = np.ascontiguousarray((image.squeeze() * 255).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    for x, y, theta, _ in minutiae:
        ix = int(round(x))
        iy = int(round(y))
        # Check image bounds
        if not (0 <= ix < W and 0 <= iy < H):
            continue

        cv2.circle(img, (ix, iy), 4, color, 2)
        angle = theta + np.pi / 2
        dx = int(15 * np.sin(angle))
        dy = int(15 * np.cos(angle))
        cv2.line(img, (ix, iy), (ix + dx, iy + dy), color, 2)
    return img.astype(np.uint8)

def draw_minutiae_comparison(image, mnt_gt, mnt_pred):
    img_gt = draw_minutiae_on_image(image, mnt_gt, color=(0, 255, 0))   # Green
    img_pred = draw_minutiae_on_image(image, mnt_pred, color=(0, 0, 255))  # Red
    concat = np.hstack([img_gt, img_pred])
    return concat.astype(np.uint8)

def draw_quality_comparision(img, quality_label, quality_pred):

    expanded_pred = np.repeat(np.repeat(quality_pred, 8, axis=1), 8, axis=0)  # (512, 512)
    expanded_label = np.repeat(np.repeat(quality_label, 8, axis=1), 8, axis=0)

    img_vis = np.ascontiguousarray((img.squeeze() * 255).astype(np.uint8))
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    q_pred_overlay = overlay_quality_map(img_vis, expanded_pred)
    q_label_overlay = overlay_quality_map(img_vis, expanded_label)

    quality_panel = np.hstack([q_label_overlay, q_pred_overlay])
    return quality_panel
