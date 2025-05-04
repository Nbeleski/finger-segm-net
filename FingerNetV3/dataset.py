import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from tools.incits378 import Template

def read_template(file):
    t = Template(file)
    return np.array([(m.x, m.y, m.orientation) for m in t.minutiae], dtype=np.float32)

def minutiae_list_to_maps(minutiae, grid_shape=(64, 64), cell_size=8, n_angle_bins=180, n_offset_bins=8):
    Hc, Wc = grid_shape
    score_map = np.zeros((1, Hc, Wc), dtype=np.float32)
    x_offset_map = np.zeros((n_offset_bins, Hc, Wc), dtype=np.float32)
    y_offset_map = np.zeros((n_offset_bins, Hc, Wc), dtype=np.float32)
    angle_map = np.zeros((n_angle_bins, Hc, Wc), dtype=np.float32)
    bin_width = 2 * np.pi / n_angle_bins
    for x, y, ang in minutiae:
        ang = ang % (2 * np.pi)  # ensure in [0, 2π)
        i = int(np.floor(y / cell_size))
        j = int(np.floor(x / cell_size))
        if 0 <= i < Hc and 0 <= j < Wc:
            score_map[0, i, j] = 1.0
            x_bin = int(round(x % cell_size))
            y_bin = int(round(y % cell_size))
            if 0 <= x_bin < n_offset_bins:
                x_offset_map[x_bin, i, j] = 1.0
            if 0 <= y_bin < n_offset_bins:
                y_offset_map[y_bin, i, j] = 1.0
            ang_bin = int(ang // bin_width) % n_angle_bins  # Bin for radians
            angle_map[ang_bin, i, j] = 1.0
    return score_map, x_offset_map, y_offset_map, angle_map

def apply_intensity_augment(img):
    """
    Apply safe grayscale intensity augmentation to a float32 NumPy image in [0, 1].
    img: np.ndarray of shape (H, W), dtype float32
    Returns a modified copy with random brightness/contrast/gamma jitter.
    """
    img = img.copy()

    if img.std() < 0.02:
        return img  # Skip low-contrast images

    transform_type = np.random.choice(["brightness", "contrast", "gamma", "none"])

    if transform_type == "brightness":
        factor = np.random.uniform(0.95, 1.05)
        img *= factor

    elif transform_type == "contrast":
        scale = np.random.uniform(0.95, 1.05)
        img = (img - 0.5) * scale + 0.5

    elif transform_type == "gamma":
        gamma = np.random.uniform(0.95, 1.05)
        img = np.power(img, gamma)

    img = np.clip(img, 0.0, 1.0)

    if img.std() < 0.01:
        return img  # Revert if augmentation collapsed contrast
    return img


class RidgeValleyDataset(Dataset):
    def __init__(self, image_dir, template_dir, seg_dir, qual_dir, filenames=None, cell_size=8,
                 n_angle_bins=180, n_offset_bins=8, augment=False):
        self.image_dir = image_dir
        self.template_dir = template_dir
        self.seg_dir = seg_dir
        self.qual_dir = qual_dir
        self.cell_size = cell_size
        self.n_angle_bins = n_angle_bins
        self.n_offset_bins = n_offset_bins

        self.filenames = filenames if filenames is not None else [
            f for f in os.listdir(image_dir) if f.lower().endswith(('.bmp', '.png', '.jpg'))
        ]
        self.filenames.sort()
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(self.image_dir, fname)
        seg_path = os.path.join(self.seg_dir, fname)
        tmpl_path = os.path.join(self.template_dir, base + '.incits378')
        qual_path = os.path.join(self.qual_dir, base + '.npy')

        # --- Load image ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        H, W = img.shape

        # --- Load segmentation and resize to match image ---
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

        # --- Load quality map
        quality = np.load(qual_path) # (64, 64)
        expanded_quality = np.repeat(np.repeat(quality, 8, axis=0), 8, axis=1)
        expanded_quality = expanded_quality.astype(np.float32) / 5.

        # --- If image is too big, randomly crop it ---
        crop_size = 512
        if H > crop_size or W > crop_size:
            top = np.random.randint(0, max(H - crop_size + 1, 1))
            left = np.random.randint(0, max(W - crop_size + 1, 1))
            img = img[top:top+crop_size, left:left+crop_size]
            seg = seg[top:top+crop_size, left:left+crop_size]
            expanded_quality = expanded_quality[top:top+crop_size, left:left+crop_size]
            crop_x, crop_y = left, top
        else:
            crop_x, crop_y = 0, 0  # no cropping

        # gotta update values, cause of the crop
        H, W = img.shape

        # --- Load and crop minutiae ---
        minutiae = []
        if os.path.exists(tmpl_path):
            for x, y, a in read_template(tmpl_path):
                x -= crop_x
                y -= crop_y
                if 0 <= x < W and 0 <= y < H:
                    minutiae.append([x, y, a])

        minutiae = np.array(minutiae)

        # --- Augmentations ---
        if self.augment:
            if random.random() < 0.5:
                img = apply_intensity_augment(img)

            # 1. Random horizontal flip
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                seg = np.fliplr(seg).copy()
                expanded_quality = np.fliplr(expanded_quality).copy()
                if len(minutiae) > 0:
                    minutiae[:, 0] = W - minutiae[:, 0] - 1
                    minutiae[:, 2] = (np.pi - minutiae[:, 2]) % (2 * np.pi)

            # 2. Random rotation [-30°, +30°]
            # angle_deg = random.uniform(-30, 30)
            angle_deg = np.random.normal(0., 12.)
            angle_rad = np.deg2rad(angle_deg)
            center = (W // 2, H // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

            #FIXME: border value?
            img = cv2.warpAffine(img, rot_mat, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.5)
            seg = cv2.warpAffine(seg, rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=1.0)
            expanded_quality = cv2.warpAffine(expanded_quality, rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.0)

            if len(minutiae) > 0:
                coords = np.stack([minutiae[:, 0], minutiae[:, 1], np.ones(len(minutiae))], axis=1)
                rotated_coords = coords @ rot_mat.T
                minutiae[:, 0:2] = rotated_coords
                minutiae[:, 2] = (minutiae[:, 2] + angle_rad) % (2 * np.pi)

                # remove minutiae that moved outside
                in_bounds = (
                    (minutiae[:, 0] >= 0) & (minutiae[:, 0] < W) &
                    (minutiae[:, 1] >= 0) & (minutiae[:, 1] < H)
                )
                minutiae = minutiae[in_bounds]

        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        seg = torch.from_numpy(seg).unsqueeze(0)  # (1, H, W)
        quality_label = expanded_quality[::8, ::8].copy()
        quality_label = torch.from_numpy(quality_label).unsqueeze(0)  # (1, H, W)
        H, W = img.shape[-2:]

        # --- Encode maps ---
        Hc, Wc = H // self.cell_size, W // self.cell_size

        score_map = np.zeros((1, Hc, Wc), dtype=np.float32)
        x_offset_map = np.zeros((self.n_offset_bins, Hc, Wc), dtype=np.float32)
        y_offset_map = np.zeros((self.n_offset_bins, Hc, Wc), dtype=np.float32)
        angle_map = np.zeros((self.n_angle_bins, Hc, Wc), dtype=np.float32)

        # --- Load minutiae ---
        if len(minutiae) > 0:
            score_map, x_offset_map, y_offset_map, angle_map = minutiae_list_to_maps(
                minutiae,
                grid_shape=(Hc, Wc),
                cell_size=self.cell_size,
                n_angle_bins=self.n_angle_bins,
                n_offset_bins=self.n_offset_bins,
            )



        # --- Masking for positive/negative in score map ---
        mnt_s_out = score_map.copy()
        seg_coarse = cv2.resize(seg.squeeze(0).numpy(), (Wc, Hc), interpolation=cv2.INTER_LINEAR)
        mask_pos = (mnt_s_out[0] == 0) & (seg_coarse > 0.3)
        mask_bg = (seg_coarse <= 0.3)
        mnt_s_out[0][mask_pos] = -1.0  # ignore
        mnt_s_out[0][mask_bg] = 0.0    # negative

        # --- To tensor ---
        return img, {
            'seg_out': seg,
            'mnt_s_out': torch.tensor(mnt_s_out, dtype=torch.float32),
            'mnt_o_out': torch.tensor(angle_map, dtype=torch.float32),
            'mnt_w_out': torch.tensor(x_offset_map, dtype=torch.float32),
            'mnt_h_out': torch.tensor(y_offset_map, dtype=torch.float32),
            'mnt_q_out': quality_label
        }
