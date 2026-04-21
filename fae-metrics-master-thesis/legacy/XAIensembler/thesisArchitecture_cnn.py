#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import metrics


# -----------------------------
#    Basic Conv Block
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_prob)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_prob)

        # Use a 1x1 conv to match dimensions for residual connection if needed.
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        if self.res_conv is not None:
            identity = self.res_conv(identity)
        x += identity
        x = F.relu(x)
        return x
# -----------------------------
#   Revised Encoder using Enhanced Conv Blocks
# -----------------------------
class SimpleEncoder(nn.Module):
    """
    CNN encoder with 3 ResidualConvBlocks and 3 pooling steps.
    """
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        # Block 1
        self.block1 = DoubleConv(in_ch, base_ch)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.block2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.block3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.block1(x)        # (B, base_ch, H, W)
        e1 = self.pool1(x1)        # (B, base_ch, H/2, W/2)

        x2 = self.block2(e1)       # (B, base_ch*2, H/2, W/2)
        e2 = self.pool2(x2)        # (B, base_ch*2, H/4, W/4)

        x3 = self.block3(e2)       # (B, base_ch*4, H/4, W/4)
        e3 = self.pool3(x3)        # (B, base_ch*4, H/8, W/8)

        return [e1, e2, e3]

# -----------------------------
#   Enhanced Cross-Attention Module
# -----------------------------
class EnhancedCrossAttention(nn.Module):
    """
    Fuses features from two streams using cross-attention.
    """
    def __init__(self, channels, attn_dropout=0.1):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.key   = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        # Scale using the dimension of the reduced query/key
        self.scale = np.sqrt(channels // 2)
        self.attn_dropout = nn.Dropout(attn_dropout)
        # Optional projection to refine the attended features
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, featA, featB):
        B, C, H, W = featA.shape
        # Generate query and key maps and reshape
        Q = self.query(featA).view(B, C // 2, -1).permute(0, 2, 1)  # (B, HW, C//2)
        K = self.key(featB).view(B, C // 2, -1)                        # (B, C//2, HW)
        # Compute attention scores and apply scaling
        attn = torch.bmm(Q, K) / self.scale                           # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # Compute value and attended output
        V = self.value(featB).view(B, C, -1).permute(0, 2, 1)           # (B, HW, C)
        out = torch.bmm(attn, V).permute(0, 2, 1).view(B, C, H, W)       # (B, C, H, W)
        out = self.out_conv(out)
        return out + featA  # Residual connection

# -----------------------------
#   Enhanced Upsampling Block
# -----------------------------
class EnhancedUpBlock(nn.Module):
    """
    Upsampling block: upsample, concatenate skip (if available), then apply ResidualConvBlock.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)    # For concatenated features
        self.conv_no_skip = DoubleConv(out_ch, out_ch)  # For cases without skip

    def forward(self, x, skip=None):
        x = self.up(x)  # Upsample

        if skip is not None:
            # In case of slight spatial mismatches, crop the skip feature map
            if x.shape[-2:] != skip.shape[-2:]:
                diffY = skip.size()[2] - x.size()[2]
                diffX = skip.size()[3] - x.size()[3]
                skip = skip[:, :, diffY // 2: diffY // 2 + x.size()[2], diffX // 2: diffX // 2 + x.size()[3]]
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
        else:
            x = self.conv_no_skip(x)
        return x

# -----------------------------
#   Final Improved Network
# -----------------------------
class EnsembleExplanationNetwork(nn.Module):
    """
    Two-stream model with cross-attention fusion at multiple scales.
    Uses deeper pooling to create smaller feature maps and enhanced blocks.
    """
    def __init__(self):
        super().__init__()
        # Separate encoders for each 3-channel explanation
        self.encoderA = SimpleEncoder(in_ch=3, base_ch=32)
        self.encoderB = SimpleEncoder(in_ch=3, base_ch=32)

        # Cross-attention modules at different scales: 32, 64, 128 channels
        self.xattn1 = EnhancedCrossAttention(channels=32)
        self.xattn2 = EnhancedCrossAttention(channels=64)
        self.xattn3 = EnhancedCrossAttention(channels=128)

        # Decoder blocks mirroring the encoder downsampling
        self.up3 = EnhancedUpBlock(128, 64)   # From H/8 -> H/4
        self.up2 = EnhancedUpBlock(64, 32)    # From H/4 -> H/2
        self.up1 = EnhancedUpBlock(32, 16)    # From H/2 -> H

        # Final 1×1 conv produces a single-channel mask; sigmoid for BCE-based segmentation
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, params, x):
        """
        x has shape (B, 6, H, W):
          - First 3 channels: model-specific explanation
          - Last 3 channels: model-agnostic explanation
        """
        # Split the input tensor into two explanations
        xA = x[:, :3, :, :]
        xB = x[:, 3:6, :, :]

        # Encode each explanation
        featsA = self.encoderA(xA)  # [e1A, e2A, e3A]
        featsB = self.encoderB(xB)  # [e1B, e2B, e3B]

        # Apply cross-attention at each scale
        f1 = self.xattn1(featsA[0], featsB[0])  # (B, 32, H/2, W/2)
        f2 = self.xattn2(featsA[1], featsB[1])  # (B, 64, H/4, W/4)
        f3 = self.xattn3(featsA[2], featsB[2])  # (B, 128, H/8, W/8)

        # Decode and progressively upsample
        d3 = self.up3(f3, f2)  # (B, 64, H/4, W/4)
        d2 = self.up2(d3, f1)  # (B, 32, H/2, W/2)
        d1 = self.up1(d2, None)  # (B, 16, H, W) -- no skip connection available here

        out = self.final_conv(d1)  # (B, 1, H, W)
        return torch.sigmoid(out), out

class MyFrame:
    def __init__(self, params, learning_rate, evalmode=False):
        self.net = EnsembleExplanationNetwork().to(params.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss = metrics.dice_bce_loss().to(params.device)
        self.lr = learning_rate
        self.params = params

    def set_input(self, img_batch, mask_batch=None):
        self.img = img_batch
        self.mask = mask_batch

    def optimize(self):
        self.optimizer.zero_grad()
        pred, _ = self.net.forward(self.params, self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss, pred

    def calculate_loss(self):
        with torch.no_grad():
            pred, ensemble_expl = self.net.forward(self.params, self.img)
            loss = self.loss(self.mask, pred)
            return loss, pred, ensemble_expl

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print("update learning rate: %f -> %f" % (self.lr, new_lr))
        self.lr = new_lr