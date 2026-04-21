import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# -----------------------
# Basic UNet Components
# -----------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # In case of mis-matched shapes due to rounding
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -----------------------
# Deeper Expert Network
# -----------------------
class DeepExpertNetwork(nn.Module):
    """
    A deeper U-Net with more down/up blocks than the original ExpertNetwork.
    Increase base_c or the depth to improve capacity.
    """

    def __init__(self, in_channels=3, base_c=16):
        super().__init__()
        # Going 4 levels down instead of 2 in the original example
        self.in_conv = DoubleConv(in_channels, base_c)  # Level 0
        self.down1 = DownBlock(base_c, base_c * 2)  # Level 1
        self.down2 = DownBlock(base_c * 2, base_c * 4)  # Level 2
        self.down3 = DownBlock(base_c * 4, base_c * 8)  # Level 3
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)  # Bottleneck

        # Up-sampling
        self.up3 = UpBlock(base_c * 16, base_c * 8)
        self.up2 = UpBlock(base_c * 8, base_c * 4)
        self.up1 = UpBlock(base_c * 4, base_c * 2)
        self.up0 = UpBlock(base_c * 2, base_c)

        self.out_conv = nn.Conv2d(base_c, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        b = self.bottleneck(x3)

        x = self.up3(b, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.out_conv(x)


# -----------------------
# Spatial Gating Network
# -----------------------
class SpatialGatingUNet(nn.Module):
    """
    A small U-Net that predicts a gating mask (num_experts channels) with the same H,W
    as the input. We can apply a softmax over the channels to get pixel-wise gating.
    """

    def __init__(self, in_channels, num_experts, base_c=8):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = DownBlock(base_c, base_c * 2)
        self.down2 = DownBlock(base_c * 2, base_c * 4)
        self.bottleneck = DoubleConv(base_c * 4, base_c * 8)

        self.up2 = UpBlock(base_c * 8, base_c * 4)
        self.up1 = UpBlock(base_c * 4, base_c * 2)
        self.up0 = UpBlock(base_c * 2, base_c)

        # Output has num_experts channels (one gating channel per expert)
        self.out_conv = nn.Conv2d(base_c, num_experts, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        b = self.bottleneck(x2)

        x = self.up2(b, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        logits = self.out_conv(x)  # (B, num_experts, H, W)
        # Convert to pixel-wise gating distributions
        gating_mask = F.softmax(logits, dim=1)  # softmax along channel dimension
        return gating_mask


# -----------------------
# Refiner UNet (Optional)
# -----------------------
class RefinerUNet(nn.Module):
    """
    Takes the pixel-wise mixture of the experts (or the stack of experts) as input
    and refines it to produce a final segmentation. Could be a small or large U-Net.
    """

    def __init__(self, in_channels=1, out_channels=1, base_c=8):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = DownBlock(base_c, base_c * 2)
        self.down2 = DownBlock(base_c * 2, base_c * 4)
        self.bottleneck = DoubleConv(base_c * 4, base_c * 8)

        self.up2 = UpBlock(base_c * 8, base_c * 4)
        self.up1 = UpBlock(base_c * 4, base_c * 2)
        self.up0 = UpBlock(base_c * 2, base_c)
        self.out_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        b = self.bottleneck(x2)

        x = self.up2(b, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        return self.out_conv(x)


# -----------------------
# MoE Ensemble
# -----------------------
class EnsembleExplanationNetwork(nn.Module):
    """
    1. We have N experts, each deeper than the original version.
    2. A spatial gating network that outputs a gating mask of shape (B, N, H, W).
    3. We do a pixel-wise combination (sum over experts with gating weights).
    4. A refinement U-Net takes the combined map (or you can also feed the stacked maps)
       as input and refines it into the final segmentation.
    """

    def __init__(self, params, channels_per_expl=3, base_c=16):
        super().__init__()
        self.params = params
        self.num_experts = len(params.XAI_methods)
        self.channels_per_expl = channels_per_expl

        # 1) Experts: each is a deeper U-Net
        self.experts = nn.ModuleList([
            DeepExpertNetwork(in_channels=channels_per_expl, base_c=base_c)
            for _ in range(self.num_experts)
        ])

        # 2) Spatial gating network: input = all explanation channels stacked
        total_in_ch = self.num_experts * channels_per_expl
        self.spatial_gate = SpatialGatingUNet(in_channels=total_in_ch,
                                              num_experts=self.num_experts,
                                              base_c=base_c // 2)  # a smaller gating net

        # 3) (Optional) A refinement UNet. Input is 1 channel (the pixel-wise mixture),
        #    or you could do more advanced combos by stacking all experts before refining.
        self.refiner = RefinerUNet(in_channels=1, out_channels=1, base_c=base_c // 2)

    def forward(self, params, x):
        """
        x shape = (B, C, H, W)
        Where C = num_experts * channels_per_expl
        """
        B, C, H, W = x.shape
        # Split input into each expert chunk
        chunked = torch.split(x, self.channels_per_expl, dim=1)  # list of length N

        # Forward pass each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            seg = expert(chunked[i])  # (B, 1, H, W) from each expert
            expert_outputs.append(seg)

        # Stack expert outputs: (B, N, H, W)
        stacked_expert_outputs = torch.cat(expert_outputs, dim=1)

        # 2) Pixel-wise gating. gating_mask: (B, N, H, W)
        gating_mask = self.spatial_gate(x)

        # 3) Weighted sum of expert outputs (pixel-wise).
        #    gating_mask[:, i, ...] multiplies expert i's output
        mixture_logits = torch.sum(stacked_expert_outputs * gating_mask, dim=1, keepdim=True)

        # 4) Refinement with a U-Net
        refined_logits = self.refiner(mixture_logits)

        # Return final sigmoid for segmentation, plus raw logits
        return torch.sigmoid(refined_logits), refined_logits


class MyFrame:
    """
    Custom training framework class.

    This class provides a custom training framework for a given neural network model.

    Args:
        net (nn.Module): Neural network model.
        learning_rate (float): Learning rate for optimization.
        device (str): Device for computations (e.g., 'cuda' or 'cpu').
        evalmode (bool): Whether to enable evaluation mode.

    Methods:
        set_input(self, img_batch, mask_batch=None): Set input data for training or evaluation.
        optimize(self): Optimize the network parameters.
        calculate_loss(self): Calculate loss and predictions.
        save(self, path): Save the model's state dictionary to a file.
        load(self, path): Load the model's state dictionary from a file.
        update_lr(self, new_lr, factor=False): Update the learning rate of the optimizer.
    """

    def __init__(self, params, learning_rate, device, evalmode=False):
        self.net = EnsembleExplanationNetwork(params).to(device)
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=learning_rate
        )
        self.loss = metrics.dice_bce_loss().to(device)
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
