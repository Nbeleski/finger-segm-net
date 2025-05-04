import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, up_in, skip_in, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in, up_in, kernel_size=2, stride=2)
        self.conv = DoubleConv(up_in + skip_in, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ShallowUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.up1 = Up(up_in=128, skip_in=64, out_channels=64)
        self.up2 = Up(up_in=64, skip_in=32, out_channels=32)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)         # (B, 32, H, W)
        x2 = self.down1(x1)      # (B, 64, H/2, W/2)
        x3 = self.down2(x2)      # (B, 128, H/4, W/4)
        x = self.up1(x3, x2)     # (B, 64, H/2, W/2)
        x = self.up2(x, x1)      # (B, 32, H, W)
        x = self.outc(x)         # (B, 1, H, W)
        return torch.sigmoid(x)

class MinutiaeFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.prelu3 = nn.PReLU(256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.prelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.prelu3(self.bn3(self.conv3(x))))
        return x

class MinutiaeHeads(nn.Module):
    def __init__(self, in_channels=256, n_bins_angle=180, n_bins_offset=8):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        self.x_offset = nn.Sequential(
            nn.Conv2d(in_channels, n_bins_offset, 1),
            nn.Sigmoid()
        )
        self.y_offset = nn.Sequential(
            nn.Conv2d(in_channels, n_bins_offset, 1),
            nn.Sigmoid()
        )
        self.angle = nn.Sequential(
            nn.Conv2d(in_channels, n_bins_angle, 1),
            nn.Sigmoid()
        )
        self.quality = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return {
            "mnt_s_score": self.score(x),
            "mnt_w_score": self.x_offset(x),
            "mnt_h_score": self.y_offset(x),
            "mnt_o_score": self.angle(x),
            "mnt_q_score": self.quality(x)
        }

class MinutiaeExtractor(nn.Module):
    def __init__(self, n_bins_angle=180, n_bins_offset=8):
        super().__init__()
        self.feature_extractor = MinutiaeFeatureExtractor(in_channels=1)
        self.heads = MinutiaeHeads(in_channels=256, n_bins_angle=n_bins_angle, n_bins_offset=n_bins_offset)

    def forward(self, seg_map):
        feats = self.feature_extractor(seg_map)
        outs = self.heads(feats)
        return outs

class FingerNetV3(nn.Module):
    def __init__(self, n_bins_angle=180, n_bins_offset=8):
        super().__init__()
        self.segmentation_net = ShallowUNet(in_channels=1, out_channels=1)
        self.minutiae_net = MinutiaeExtractor(n_bins_angle=n_bins_angle, n_bins_offset=n_bins_offset)

    def forward(self, x):  # x: (B, 1, H, W) grayscale image
        seg = self.segmentation_net(x)           # (B, 1, H, W), raw sigmoid
        minutiae = self.minutiae_net(seg)        # dict of score, x_offset, y_offset, angle
        return {
            'segmentation': seg,
            'minutiae': minutiae
        }
    
        # return seg, mnt["score"], mnt["x_offset"], mnt["y_offset"], mnt["angle"]
        
