import torch
import torch.nn as nn
import torch.nn.functional as F
import metrics

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_prob)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_prob)

        # Residual dimension adjustment if needed
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

# ---------------------------
# EnhancedEncoder (unchanged)
# ---------------------------
class EnhancedEncoder(nn.Module):
    """
    3 downsamplings -> final feature, plus skip connections for U-Net style decoding.
    """
    def __init__(self, in_ch, base_ch=32, dropout=0.1):
        super().__init__()
        # Level 1
        self.block1 = DoubleConv(in_ch, base_ch, dropout_prob=dropout)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 2
        self.block2 = DoubleConv(base_ch, base_ch*2, dropout_prob=dropout)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 3
        self.block3 = DoubleConv(base_ch*2, base_ch*4, dropout_prob=dropout)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)

        # "Deep" block
        self.block4 = DoubleConv(base_ch*4, base_ch*4, dropout_prob=dropout)

        self.out_channels = base_ch*4

    def forward(self, x):
        # 1) Downsample
        x1 = self.block1(x)      # (B, base_ch,   H,   W)
        x  = self.pool1(x1)      # (B, base_ch,   H/2, W/2)

        # 2) Downsample
        x2 = self.block2(x)      # (B, base_ch*2, H/2, W/2)
        x  = self.pool2(x2)      # (B, base_ch*2, H/4, W/4)

        # 3) Downsample
        x3 = self.block3(x)      # (B, base_ch*4, H/4, W/4)
        x  = self.pool3(x3)      # (B, base_ch*4, H/8, W/8)

        # 4) Deep block
        x4 = self.block4(x)      # (B, base_ch*4, H/8, W/8)

        # Return the deep feature plus the skip features
        return x4, (x1, x2, x3)

# ---------------------------
# EnhancedDecoder (unchanged)
# ---------------------------
class EnhancedDecoder(nn.Module):
    """
    3 upsamplings (mirroring the 3 in the encoder).
    Skip connections used from x1, x2, x3.
    """
    def __init__(self, out_ch=1, base_ch=32, dropout=0.1):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*4, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch*4 + base_ch*4, base_ch*4, dropout_prob=dropout)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch*2 + base_ch*2, base_ch*2, dropout_prob=dropout)

        self.up3 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch + base_ch, base_ch, dropout_prob=dropout)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x4, skips):
        # x4 is (B, base_ch*4, H/8, W/8)
        # skips is (x1, x2, x3)
        x1, x2, x3 = skips

        # Up 1
        x = self.up1(x4)               # (B, base_ch*4, H/4, W/4)
        x = torch.cat([x, x3], dim=1)  # skip from x3
        x = self.conv1(x)              # (B, base_ch*4, H/4, W/4)

        # Up 2
        x = self.up2(x)               # (B, base_ch*2, H/2, W/2)
        x = torch.cat([x, x2], dim=1) # skip from x2
        x = self.conv2(x)             # (B, base_ch*2, H/2, W/2)

        # Up 3
        x = self.up3(x)               # (B, base_ch,   H,   W)
        x = torch.cat([x, x1], dim=1) # skip from x1
        x = self.conv3(x)             # (B, base_ch,   H,   W)

        logits = self.out_conv(x)     # (B, out_ch, H, W)
        return logits

# ---------------------------
# GatedFusion (unchanged for the deep features)
# ---------------------------
class GatedFusion(nn.Module):
    """
    Gating net that outputs alpha_i per expert at each spatial location.
    1) Stack [F1, F2, ..., Fk] along channel dimension => (B, k*C, H, W)
    2) Tiny gating subnetwork => (B, k, H, W)
    3) Softmax across k => alpha_i
    4) Weighted sum => fused
    """
    def __init__(self, in_ch_each, num_experts, base_ch_gate=64):
        super().__init__()
        self.num_experts = num_experts
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_ch_each * num_experts, base_ch_gate, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch_gate),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch_gate, num_experts, kernel_size=3, padding=1)
        )

    def forward(self, features_list):
        # (B, C, H, W) * num_experts => stack on channel
        fused_cat = torch.cat(features_list, dim=1)  # (B, num_experts*C, H, W)
        logits = self.gate_net(fused_cat)            # (B, num_experts,   H, W)
        alpha = F.softmax(logits, dim=1)             # (B, num_experts,   H, W)

        # Weighted sum
        fused = torch.zeros_like(features_list[0])
        for i in range(self.num_experts):
            fused += alpha[:, i:i+1, :, :] * features_list[i]

        return fused, alpha

# ---------------------------
# EnsembleExplanationNetwork
# ---------------------------
class EnsembleExplanationNetwork(nn.Module):
    """
    Gated fusion network with N=number_of_explanations encoders.
    Each encoder => EnhancedEncoder
    GatedFusion => combine deep features from each encoder
    Single EnhancedDecoder => final mask
    Optionally, fuse skip connections from all encoders at each level via GatedFusion.
    """
    def __init__(self,
                 params,
                 channels_per_expl=3,
                 base_ch=32,
                 dropout=0.1,
                 fuse_skips=True):
        super().__init__()
        self.num_explanations  = len(params.XAI_methods)
        self.channels_per_expl = channels_per_expl
        self.fuse_skips        = fuse_skips

        # Create separate encoders
        self.encoders = nn.ModuleList([
            EnhancedEncoder(in_ch=channels_per_expl, base_ch=base_ch, dropout=dropout)
            for _ in range(self.num_explanations)
        ])

        # Gated fusion to combine deep features from each encoder
        # Each encoder output is (base_ch*4) at the final layer
        self.gated_fusion_deep = GatedFusion(
            in_ch_each=base_ch*4,
            num_experts=self.num_explanations,
            base_ch_gate=64
        )

        # If we also want to fuse skip connections from all encoders:
        # The skip levels have channels: [base_ch, 2*base_ch, 4*base_ch].
        # We create a GatedFusion for each skip level.
        if self.fuse_skips:
            self.gated_fusion_skips = nn.ModuleList([
                GatedFusion(in_ch_each=base_ch,     num_experts=self.num_explanations, base_ch_gate=base_ch),
                GatedFusion(in_ch_each=base_ch*2,   num_experts=self.num_explanations, base_ch_gate=base_ch*2),
                GatedFusion(in_ch_each=base_ch*4,   num_experts=self.num_explanations, base_ch_gate=base_ch*4),
            ])
        else:
            self.gated_fusion_skips = None

        # Decoder: The fused deep feature has shape (B, base_ch*4, H/8, W/8)
        self.decoder = EnhancedDecoder(out_ch=1, base_ch=base_ch, dropout=dropout)

    def forward(self, params, x):
        """
        x shape: (B, 3 * num_explanations, H, W)
        Returns:
          mask:   (B,1,H,W) - after sigmoid
          logits: (B,1,H,W) - raw
        """
        B, C, H, W = x.shape
        expected = self.num_explanations * self.channels_per_expl
        assert C == expected, f"Expected {expected} channels, got {C}"

        # 1) Split input by explanation => list of shape (B, channels_per_expl, H, W)
        chunk_size = self.channels_per_expl
        expl_chunks = torch.split(x, chunk_size, dim=1)

        # 2) Encode each explanation
        deep_feats = []
        all_skips  = []
        for i, encoder in enumerate(self.encoders):
            deep, skips = encoder(expl_chunks[i])
            deep_feats.append(deep)
            all_skips.append(skips)  # each 'skips' is (x1, x2, x3)

        # 3) Gated fusion across these deep features
        fused_deep, alpha_deep = self.gated_fusion_deep(deep_feats)
        # fused_deep: (B, base_ch*4, H/8, W/8)

        if self.fuse_skips:
            # For each skip level, fuse across encoders
            # We have 3 skip levels: x1, x2, x3 for each encoder
            # We'll gather them and pass them to GatedFusion
            skip1_list = [sk[0] for sk in all_skips]  # each is (B, base_ch, H, W)
            skip2_list = [sk[1] for sk in all_skips]  # (B, 2*base_ch, H/2, W/2)
            skip3_list = [sk[2] for sk in all_skips]  # (B, 4*base_ch, H/4, W/4)

            fused_skip1, _ = self.gated_fusion_skips[0](skip1_list)
            fused_skip2, _ = self.gated_fusion_skips[1](skip2_list)
            fused_skip3, _ = self.gated_fusion_skips[2](skip3_list)

            chosen_skips = (fused_skip1, fused_skip2, fused_skip3)
        else:
            # original approach: use skip connections from the *first* encoder
            chosen_skips = all_skips[0]

        # 4) Decode fused deep feature -> final mask
        logits = self.decoder(fused_deep, chosen_skips)  # shape (B,1,H,W)
        mask   = torch.sigmoid(logits)

        return mask, logits

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

    def __init__(self, params, learning_rate, evalmode=False):
        self.net = EnsembleExplanationNetwork(params).to(params.device)
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=learning_rate
        )
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
