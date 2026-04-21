import torch
import torch.nn as nn
import torch.nn.functional as F
import metrics

# -----------------------------
#  Squeeze-and-Excitation (SE) Block
#  (Channel-wise attention)
# -----------------------------
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block applies global average pooling to squeeze the feature maps
    and then applies two linear layers to learn channel-wise excitation.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        # Squeeze (Global Average Pool)
        y = self.avg_pool(x).view(b, c)  # (B, C)
        # Excitation
        y = self.fc1(y)                  # (B, C//reduction)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)                  # (B, C)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -----------------------------
#  Attention Gate for skip connections (spatial attention)
#  Based on: "Attention U-Net: Learning Where to Look for the Pancreas" (Oktay et al.)
# -----------------------------
class AttentionGate(nn.Module):
    """
    G(x, g) -> Attentional mask * x
    x: skip connection feature
    g: gating signal (coarser feature)
    """
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

    def forward(self, x, g):
        """
        x: skip connection feature map (B, in_channels_x, H, W)
        g: gating feature map (B, in_channels_g, H', W')
        """
        # Bring gating to the same spatial shape as x:
        # Typically we upsample 'g' if it's smaller.
        # For a 2-level UNet, g might be x's size or smaller.
        g_upsampled = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        Wx = self.Wx(x)  # (B, inter_channels, H, W)
        Wg = self.Wg(g_upsampled)  # (B, inter_channels, H, W)

        # element-wise sum, then relu
        psi = F.relu(Wx + Wg, inplace=True)
        psi = self.psi(psi)  # (B, 1, H, W)
        alpha = torch.sigmoid(psi)  # (B, 1, H, W)

        # output
        return x * alpha

# -----------------------------
#  ASPP (Atrous Spatial Pyramid Pooling)
#  for better multi-scale context at the bottleneck
# -----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=(1, 6, 12, 18)):
        """
        atrous_rates: dilation rates
        """
        super().__init__()
        self.convs = nn.ModuleList()
        for rate in atrous_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate, padding=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # 1x1 conv branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        res = [conv(x) for conv in self.convs]  # multiple dilated convs
        res.append(self.conv1x1(x))            # 1x1 conv branch
        out = torch.cat(res, dim=1)            # fuse
        out = self.project(out)
        return out

# -----------------------------
#  A "DoubleConv" block with residual + SE block
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1, use_se=True):
        super().__init__()
        self.use_se = use_se

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

        if self.use_se:
            self.se_block = SEBlock(out_channels)

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

        # Apply SE block (channel attention)
        if self.use_se:
            x = self.se_block(x)

        x = F.relu(x)
        return x

# -----------------------------
#  Encoder (2-level) with DoubleConv + ASPP in the bottleneck
# -----------------------------
class Encoder2Level(nn.Module):
    def __init__(self, in_channels, base_channels=32, dropout_prob=0.1, use_se=True):
        super().__init__()
        # Block 1: no pooling yet (skip1)
        self.block1 = DoubleConv(in_channels, base_channels, dropout_prob, use_se=use_se)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: skip2-level
        self.block2 = DoubleConv(base_channels, base_channels * 2, dropout_prob, use_se=use_se)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ASPP in the deep/bottleneck
        self.aspp = ASPP(base_channels * 2, base_channels * 2, atrous_rates=(1, 6, 12, 18))

        # Output channel info
        self.out_channels_skip1 = base_channels
        self.out_channels_skip2 = base_channels * 2
        self.out_channels_deep = base_channels * 2

    def forward(self, x):
        skip1 = self.block1(x)             # (B, base_channels, H, W)
        x = self.pool1(skip1)              # (B, base_channels, H/2, W/2)

        skip2 = self.block2(x)             # (B, base_channels*2, H/2, W/2)
        x = self.pool2(skip2)              # (B, base_channels*2, H/4, W/4)

        deep = self.aspp(x)                # (B, base_channels*2, H/4, W/4)

        return skip1, skip2, deep

# -----------------------------
#  Decoder with Attention Gates + improved blocks
# -----------------------------
class ImprovedDecoder2Level(nn.Module):
    def __init__(
        self,
        in_channels_deep,
        in_channels_skip2,
        in_channels_skip1,
        base_channels=32,
        dropout_prob=0.1,
        use_se=True
    ):
        super().__init__()

        # Attention gates to refine skip connections
        self.attention_gate2 = AttentionGate(
            in_channels_x=in_channels_skip2,  # skip2
            in_channels_g=in_channels_deep,   # gating from deep
            inter_channels=in_channels_skip2 // 2
        )
        self.attention_gate1 = AttentionGate(
            in_channels_x=in_channels_skip1,  # skip1
            in_channels_g=base_channels * 2,  # gating from the upsampled feature after skip2 merge
            inter_channels=in_channels_skip1 // 2
        )

        # Upsample deep features and fuse with skip2
        self.up2 = nn.ConvTranspose2d(in_channels_deep, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 2 + in_channels_skip2, base_channels * 2, dropout_prob, use_se=use_se)

        # Upsample and fuse with skip1
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels + in_channels_skip1, base_channels, dropout_prob, use_se=use_se)

        # Final 1×1 conv to generate output mask
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, deep, skip2, skip1):
        # Gate skip2 using deep features
        skip2_att = self.attention_gate2(skip2, deep)
        x = self.up2(deep)  # Upsample to H/2 resolution
        x = torch.cat([x, skip2_att], dim=1)
        x = self.conv2(x)

        # Gate skip1 using new intermediate feature x
        skip1_att = self.attention_gate1(skip1, x)
        x = self.up1(x)  # Upsample to full resolution
        x = torch.cat([x, skip1_att], dim=1)
        x = self.conv1(x)

        out = self.out_conv(x)
        return out

# -----------------------------
#  Ensemble Explanation Network (Late Fusion) with
#  - separate encoders for each explanation input
#  - multi-scale feature fusion
#  - single decoder with attention gates
# -----------------------------
class EnsembleExplanationNetwork(nn.Module):
    def __init__(self, params, channels_per_expl=3, base_channels=32, dropout_prob=0.1, use_se=True):
        """
        params: object that contains 'XAI_methods', a list of explanation methods
        channels_per_expl: number of channels per explanation input (e.g., 3 for RGB)
        """
        super().__init__()
        self.num_explanations = len(params.XAI_methods)
        self.channels_per_expl = channels_per_expl

        # Create one encoder per explanation method
        self.encoders = nn.ModuleList([
            Encoder2Level(
                in_channels=channels_per_expl,
                base_channels=base_channels,
                dropout_prob=dropout_prob,
                use_se=use_se
            )
            for _ in range(self.num_explanations)
        ])

        # Fusion blocks to merge multi-scale features from different explanation branches
        self.fusion_skip1 = nn.Conv2d(base_channels * self.num_explanations, base_channels, kernel_size=1)
        self.fusion_skip2 = nn.Conv2d((base_channels * 2) * self.num_explanations, base_channels * 2, kernel_size=1)
        self.fusion_deep = nn.Conv2d((base_channels * 2) * self.num_explanations, base_channels * 2, kernel_size=1)

        # Single decoder that accepts the fused features
        self.decoder = ImprovedDecoder2Level(
            in_channels_deep=base_channels * 2,  # after fusion, deep has base_channels*2
            in_channels_skip2=base_channels * 2, # after fusion, skip2 has base_channels*2
            in_channels_skip1=base_channels,     # after fusion, skip1 has base_channels
            base_channels=base_channels,
            dropout_prob=dropout_prob,
            use_se=use_se
        )

    def forward(self, params, x):
        """
        x: Tensor of shape (B, channels_per_expl * num_explanations, H, W)
           e.g., with 3 explanation methods & RGB: (B, 9, H, W)
        """
        B, C, H, W = x.shape
        expected_channels = self.channels_per_expl * self.num_explanations
        assert C == expected_channels, f"Expected {expected_channels} channels, got {C}."

        # Split the input into separate explanation branches
        chunks = torch.split(x, self.channels_per_expl, dim=1)
        skip1_list, skip2_list, deep_list = [], [], []

        for i, encoder in enumerate(self.encoders):
            s1, s2, d = encoder(chunks[i])
            skip1_list.append(s1)
            skip2_list.append(s2)
            deep_list.append(d)

        # Concatenate features from all branches along the channel dimension
        fused_skip1 = torch.cat(skip1_list, dim=1)  # (B, base_channels * num_explanations, H, W)
        fused_skip2 = torch.cat(skip2_list, dim=1)  # (B, base_channels*2 * num_explanations, H/2, W/2)
        fused_deep = torch.cat(deep_list, dim=1)    # (B, base_channels*2 * num_explanations, H/4, W/4)

        # Apply 1x1 conv fusion
        fused_skip1 = self.fusion_skip1(fused_skip1)  # (B, base_channels, H, W)
        fused_skip2 = self.fusion_skip2(fused_skip2)  # (B, base_channels*2, H/2, W/2)
        fused_deep = self.fusion_deep(fused_deep)     # (B, base_channels*2, H/4, W/4)

        # Decode the fused features
        logits = self.decoder(fused_deep, fused_skip2, fused_skip1)
        return torch.sigmoid(logits), logits


class MyFrame:
    def __init__(self, params, learning_rate, evalmode=False):
        self.net = EnsembleExplanationNetwork(params).to(params.device)
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
