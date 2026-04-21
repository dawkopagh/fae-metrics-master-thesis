import torch
import torch.nn as nn
import torch.nn.functional as F
import metrics
import math

# -----------------------------
# 1) Patch + Position Embedding
# -----------------------------
class PatchEmbedding(nn.Module):
    """
    Splits a (B, in_channels, H, W) image into patches of size (patch_size, patch_size).
    Each patch is flattened into a vector, which is projected to a given embedding dimension.
    """
    def __init__(self, in_channels, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2

        # Linear proj for flattened patches:
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # Positional embeddings:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        x shape: (B, in_channels, H, W)
        Returns patch embeddings of shape: (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size  # Patch grid size

        # Reshape input into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patches_h, num_patches_w, C, patch_size, patch_size)

        # Flatten patches
        patches = patches.view(B, ph * pw, C * self.patch_size * self.patch_size)  # (B, num_patches, patch_dim)

        # Project to embedding space
        x = self.proj(patches)  # (B, num_patches, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        return x


# -----------------------------
# 2) Transformer Encoder Blocks
# -----------------------------
class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block: MHSA + MLP (feed-forward).
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Layer Norm
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP / feed-forward
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, num_patches, embed_dim)
        # 1) Self-Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # shape same as x
        x = x + self.dropout(attn_out)

        # 2) Feed Forward
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)
        return x


class TransformerEncoder(nn.Module):
    """
    A stack of N Transformer blocks.
    """
    def __init__(self, embed_dim=768, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------
# 3) Vision Transformer Segmenter
# -----------------------------
class EnsembleExplanationNetwork(nn.Module):
    """
    A minimal Vision Transformer for segmentation:
      - Patchify input (3 * params.XAI_methods channels)
      - Transformer encoder
      - Reshape to 2D, upsample, project to 1-channel mask
    """
    def __init__(
        self,
        params=None,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=8,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.params = params
        self.in_channels = 3 * len(params.XAI_methods)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = 1

        self.patch_embed = PatchEmbedding(
            in_channels=self.in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # Final segmentation head:
        # 1) Project from embed_dim -> smaller feature dimension
        # 2) Reshape to (B, some_feat, #patches_y, #patches_x)
        # 3) Upsample to (B, some_feat, H, W)
        # 4) Final 1×1 conv -> 1 channel => Sigmoid
        self.head_channels = 256  # intermediate dimension for segmentation
        self.proj = nn.Linear(embed_dim, self.head_channels)
        self.upconv = nn.Conv2d(self.head_channels, self.num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, params, x):
        """
        x shape: (B, in_channels, 224, 224)
        Returns:
         - pred:  (B,1,224,224)  final mask
         - ensemble_expl: same as pred or any intermediate representation
        """
        B, C, H, W = x.shape

        # 1) Patchify + position embed => (B, num_patches, embed_dim)
        tokens = self.patch_embed(x)

        # 2) Transformer encoder => (B, num_patches, embed_dim)
        encoded = self.transformer(tokens)

        # 3) Project to head_channels => (B, num_patches, head_channels)
        encoded = self.proj(encoded)

        # 4) Reshape to 2D feature map
        patches_h = H // self.patch_size
        patches_w = W // self.patch_size
        encoded = encoded.transpose(1, 2)  # (B, head_channels, num_patches)
        encoded = encoded.view(B, self.head_channels, patches_h, patches_w)  # (B,head_channels,14,14) if patch=16

        # 5) Upsample to original resolution => (B, head_channels, H, W)
        encoded = F.interpolate(encoded, size=(H, W), mode="bilinear", align_corners=False)

        # 6) Final 1x1 conv -> (B, 1, H, W)
        out = self.upconv(encoded)

        # 7) Sigmoid for single-channel output
        pred = self.sigmoid(out)

        # Return single-channel mask
        ensemble_expl = pred  # Or some other intermediate if you wish
        return pred, ensemble_expl

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
