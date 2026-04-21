import torch
import torch.nn as nn
import torch.nn.functional as F
import metrics

# -----------------------------------------------------
# A) Gating block (optional)
# -----------------------------------------------------
class ExplanationGate(nn.Module):
    """
    A simple gating mechanism that learns per-explanation weighting.
    For 2 explanations × 3 channels = 6 channels total, we do a
    1×1 conv -> sigmoid to get gating maps, then multiply.
    If `gating_mode='per_channel'`, we gate each channel individually.
    If `gating_mode='shared_explanation'`, we can produce a single
    gating weight per explanation, etc.
    """
    def __init__(self, in_ch, gating_mode='per_channel'):
        super().__init__()
        self.gating_mode = gating_mode
        # Example: default to gating per channel
        self.gate_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        """
        x shape: (B, in_ch, H, W)
        Returns: gated x of the same shape
        """
        gate = self.sigmoid(self.gate_conv(x))
        return x * gate


# -----------------------------------------------------
# B) CNN Encoder (ResNet-like)
#     - returns multiple scale features for a U-Net style decoder
# -----------------------------------------------------
class BasicBlock(nn.Module):
    """
    A small residual block: Conv -> BN -> ReLU -> Conv -> BN, skip connect.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # If shape changes (due to stride or channel mismatch), use 1x1 conv for skip
        self.skip  = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip:
            identity = self.skip(x)
        out += identity
        out = self.relu(out)
        return out


class CNNEncoder(nn.Module):
    """
    Simple ResNet-like encoder that produces feature maps at multiple scales:
    x1: 1/2 resolution
    x2: 1/4 resolution
    x3: 1/8 resolution
    x4: 1/16 resolution (bottleneck)
    """
    def __init__(self, in_ch=6, base_ch=64):
        super().__init__()
        # Stage 0: initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        # Stage 1: down to 1/2
        self.layer1 = BasicBlock(base_ch, base_ch, stride=1)  # still 1/2
        # Stage 2: down to 1/4
        self.layer2 = BasicBlock(base_ch, base_ch*2, stride=2)
        # Stage 3: down to 1/8
        self.layer3 = BasicBlock(base_ch*2, base_ch*4, stride=2)
        # Stage 4: down to 1/16
        self.layer4 = BasicBlock(base_ch*4, base_ch*8, stride=2)

    def forward(self, x):
        # Input shape: (B, in_ch, 224, 224)
        x0 = self.stem(x)                # (B, base_ch,   112,112)
        x1 = self.layer1(x0)             # (B, base_ch,   112,112) [1/2]
        x2 = self.layer2(x1)             # (B, base_ch*2, 56,56)   [1/4]
        x3 = self.layer3(x2)             # (B, base_ch*4, 28,28)   [1/8]
        x4 = self.layer4(x3)             # (B, base_ch*8, 14,14)   [1/16]
        return x1, x2, x3, x4


# -----------------------------------------------------
# C) Transformer Encoder + Learnable 2D position embeddings
# -----------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0.0):
        """
        Standard Transformer block:
          x -> LN -> MultiheadAttn -> Drop -> residual ->
               LN -> MLP -> Drop -> residual
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)

        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        # x shape: (B, N, D)
        res = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = res + self.drop(x)

        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = res + self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Stacks multiple TransformerBlocks, plus a learnable position embedding.
    """
    def __init__(self, embed_dim=256, num_heads=4, depth=4, mlp_ratio=4.0, drop_rate=0.0, num_tokens=14*14):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x shape: (B, N, D), where N = 14×14 from the last encoder feature
        B, N, D = x.shape
        # Add position embedding
        x = x + self.pos_emb[:, :N, :]  # shape (1, N, D) is broadcast over B
        for blk in self.blocks:
            x = blk(x)
        return x


# -----------------------------------------------------
# D) U-Net Style Decoder
# -----------------------------------------------------
class DecoderBlock(nn.Module):
    """
    A single decoder block that upscales the feature map by 2,
    merges with a skip connection, and does a residual conv block.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = BasicBlock(out_ch + skip_ch, out_ch, stride=1)

    def forward(self, x, skip):
        x = self.up(x)             # upsample by 2
        x = torch.cat([x, skip], dim=1)  # concatenate skip
        x = self.conv(x)
        return x


# -----------------------------------------------------
# E) Full Model
# -----------------------------------------------------
class EnsembleExplanationNetwork(nn.Module):
    """
    Combines:
      1) Optional gating for multi-explanation input
      2) A ResNet-like CNN encoder
      3) A Transformer bottleneck
      4) A U-Net style decoder
    """
    def __init__(
        self,
        params,
        in_ch=6,           # e.g. 2 explanations × 3 channels
        base_ch=64,
        embed_dim=256,
        num_heads=4,
        depth=4,
        drop_rate=0.1,
        out_ch=1,
        use_gating=True
    ):
        super().__init__()
        self.params = params

        self.use_gating = use_gating
        if use_gating:
            self.gate = ExplanationGate(in_ch=in_ch, gating_mode='per_channel')
        else:
            self.gate = nn.Identity()

        # CNN encoder
        self.encoder = CNNEncoder(in_ch=in_ch, base_ch=base_ch)

        # Transformer bottleneck at 1/16 scale
        self.transformer_dim = embed_dim
        self.bottleneck_in_ch = base_ch * 8

        # Project encoder feature -> transformer dim
        self.bottleneck_conv = nn.Conv2d(self.bottleneck_in_ch, embed_dim, kernel_size=1)

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=4.0,
            drop_rate=drop_rate,
            num_tokens=14*14
        )

        # Project back from transformer dim -> CNN channels
        self.bottleneck_deconv = nn.Conv2d(embed_dim, self.bottleneck_in_ch, kernel_size=1)

        # Decoder
        self.dec3 = DecoderBlock(in_ch=base_ch*8, skip_ch=base_ch*4, out_ch=base_ch*4)
        self.dec2 = DecoderBlock(in_ch=base_ch*4, skip_ch=base_ch*2, out_ch=base_ch*2)
        self.dec1 = DecoderBlock(in_ch=base_ch*2, skip_ch=base_ch,   out_ch=base_ch)

        # Final upsample + 1×1 conv for the output mask
        self.up0       = nn.ConvTranspose2d(base_ch, base_ch//2, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(base_ch//2, out_ch, kernel_size=3, padding=1)

    def forward(self, params, x):
        """
        x shape: (B, in_ch, 224, 224)
        Returns: predicted mask of shape (B, out_ch, 224, 224)
        """
        # 1) Gating
        x = self.gate(x)

        # 2) CNN Encoder
        x1, x2, x3, x4 = self.encoder(x)
        # shapes:
        #   x1: (B, base_ch,   112,112) [1/2]
        #   x2: (B, base_ch*2, 56,56)   [1/4]
        #   x3: (B, base_ch*4, 28,28)   [1/8]
        #   x4: (B, base_ch*8, 14,14)   [1/16]

        # 3) Transformer bottleneck
        bottleneck = self.bottleneck_conv(x4)  # => (B, embed_dim, 14,14)

        # Flatten to (B, N, D) for the transformer
        B, D, H, W = bottleneck.shape
        bottleneck_reshaped = bottleneck.flatten(2).permute(0, 2, 1)  # (B, HW=14*14, D=embed_dim)

        # Pass through Transformer
        b_feat = self.transformer(bottleneck_reshaped)                # => (B, 14*14, embed_dim)

        # Reshape back
        b_feat = b_feat.permute(0, 2, 1).view(B, D, H, W)
        bottleneck = self.bottleneck_deconv(b_feat)   # => (B, base_ch*8, 14,14)

        # 4) U-Net Decoder
        d3 = self.dec3(bottleneck, x3)   # => (B, base_ch*4, 28,28)
        d2 = self.dec2(d3, x2)          # => (B, base_ch*2, 56,56)
        d1 = self.dec1(d2, x1)          # => (B, base_ch,   112,112)

        up0 = self.up0(d1)              # => (B, base_ch//2, 224,224)
        out = self.final_conv(up0)      # => (B, out_ch, 224,224)

        return torch.sigmoid(out), out

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
