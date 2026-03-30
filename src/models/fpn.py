"""
Simple Feature Pyramid Network
"""
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style Simple Feature Pyramid.

    Takes 3 feature maps from UNI (all [B, 1024, 14, 14]) and produces
    multi-scale outputs:
        P2: [B, D2, 28, 28]  <- 1x deconv upsample from block 11
        P3: [B, D3, 14, 14]  <- identity projection from block 17
        P4: [B, D4,  7,  7]  <- stride-2 pool from block 23
    """
    def __init__(self, in_dim=1024, out_dims=(128, 320, 512)):
        super().__init__()
        D2, D3, D4 = out_dims

        # P2: 14 -> 28  (1x deconv upsample)
        self.up2 = nn.Sequential(
            nn.Conv2d(in_dim, D2, 1, bias=False), nn.BatchNorm2d(D2), nn.GELU(),
            nn.ConvTranspose2d(D2, D2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(D2), nn.GELU(),
            nn.Conv2d(D2, D2, 3, padding=1, bias=False), nn.BatchNorm2d(D2), nn.GELU(),
        )
        # P3: 14 -> 14  (identity + channel projection)
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_dim, D3, 1, bias=False), nn.BatchNorm2d(D3), nn.GELU(),
            nn.Conv2d(D3, D3, 3, padding=1, bias=False), nn.BatchNorm2d(D3), nn.GELU(),
        )
        # P4: 14 -> 7  (stride-2 pool + channel projection)
        self.down4 = nn.Sequential(
            nn.Conv2d(in_dim, D4, 1, bias=False), nn.BatchNorm2d(D4), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(D4, D4, 3, padding=1, bias=False), nn.BatchNorm2d(D4), nn.GELU(),
        )

    def forward(self, feats):
        f11, f17, f23 = feats   # all [B, 1024, 14, 14]
        p2 = self.up2(f11)          # [B, D2, 28, 28]
        p3 = self.proj3(f17)        # [B, D3, 14, 14]
        p4 = self.down4(f23)        # [B, D4,  7,  7]
        return p2, p3, p4
