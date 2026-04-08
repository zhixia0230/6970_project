"""
Main Model v8b: IR-50 + Landmark-Enhanced CBAM + Self-Attention

Key change vs v8: instead of a heavy cross-attention module,
we inject landmark heatmaps directly into CBAM's spatial attention
as a soft prior. Zero new parameters — just a bias signal.

Spatial attention becomes: sigmoid(conv(cat(avg, max)) + landmark_prior)
instead of:                sigmoid(conv(cat(avg, max)))

This tells CBAM "eyes and mouth are important" without adding
complexity that overfits on 12K samples.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED_ONNX = os.path.join(os.path.dirname(__file__),
                               'pretrained', 'w600k_r50.onnx')


# ─── Landmark Heatmap ──────────────────────────────────────────────────────────

class LandmarkHeatmap(nn.Module):
    """Generate soft Gaussian heatmaps from 5 landmarks. No learnable params."""
    def __init__(self, sigma=1.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, landmarks, feat_h, feat_w, img_size=112):
        """landmarks: (B, 5, 2), returns: (B, 1, feat_h, feat_w)"""
        B = landmarks.shape[0]
        device = landmarks.device
        scale_x = feat_w / img_size
        scale_y = feat_h / img_size
        lm_x = landmarks[:, :, 0] * scale_x
        lm_y = landmarks[:, :, 1] * scale_y

        yy = torch.arange(feat_h, device=device, dtype=torch.float32)
        xx = torch.arange(feat_w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        gx = gx.unsqueeze(0).unsqueeze(0)
        gy = gy.unsqueeze(0).unsqueeze(0)
        cx = lm_x.unsqueeze(-1).unsqueeze(-1)
        cy = lm_y.unsqueeze(-1).unsqueeze(-1)

        # Sum of 5 Gaussian blobs → single heatmap
        heatmaps = torch.exp(
            -((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * self.sigma ** 2)
        )  # (B, 5, H, W)
        combined = heatmaps.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        return combined


# ─── CBAM with Landmark Prior ──────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(x.mean(dim=[2, 3]))
        mx = self.fc(x.amax(dim=[2, 3]))
        return x * torch.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttentionWithLandmark(nn.Module):
    """
    Standard CBAM spatial attention + optional landmark heatmap prior.
    When landmarks are provided, the heatmap is added to the attention logits
    BEFORE sigmoid, biasing attention toward eyes/nose/mouth.
    """
    def __init__(self, kernel_size=7, landmark_weight=0.5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.landmark_weight = landmark_weight

    def forward(self, x, landmark_heatmap=None):
        desc = torch.cat([x.mean(1, keepdim=True),
                          x.amax(1, keepdim=True)], dim=1)
        attn_logits = self.conv(desc)  # (B, 1, H, W)

        if landmark_heatmap is not None:
            attn_logits = attn_logits + self.landmark_weight * landmark_heatmap

        return x * torch.sigmoid(attn_logits)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, landmark_weight=0.5):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttentionWithLandmark(7, landmark_weight)

    def forward(self, x, landmark_heatmap=None):
        out = self.ca(x)
        out = self.sa(out, landmark_heatmap)
        return out + x


# ─── Self-Attention ────────────────────────────────────────────────────────────

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        tokens = x.flatten(2).transpose(1, 2)

        t = self.norm1(tokens)
        qkv = self.qkv(t).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.drop((q @ k.transpose(-2, -1) * self.scale).softmax(-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        tokens = tokens + self.drop(self.proj(out))
        tokens = tokens + self.ffn(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ─── IR-50 Backbone ────────────────────────────────────────────────────────────

class IR50Backbone(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        import onnx
        import onnx2torch
        self.ir50 = onnx2torch.convert(onnx.load(onnx_path))
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def fn(module, inp, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    self._features[name] = output
            return fn

        for name, mod in self.ir50.named_children():
            if name == 'BatchNormalization_110':
                mod.register_forward_hook(hook_fn('stage3'))
            elif name == 'BatchNormalization_126':
                mod.register_forward_hook(hook_fn('stage4'))

    def forward(self, x):
        self._features.clear()
        _ = self.ir50(x)
        return self._features.get('stage3'), self._features.get('stage4')


# ─── Main Model ───────────────────────────────────────────────────────────────

class EmotionCNNAttention(nn.Module):
    """
    IR-50 + Landmark-Enhanced CBAM + Multi-Scale Fusion + Self-Attention

    Same structure as v7b, but CBAM spatial attention is biased by
    landmark heatmaps (zero extra parameters).
    """
    def __init__(self, num_classes=7, dropout=0.4, pretrained=True,
                 landmark_weight=0.5):
        super().__init__()

        if pretrained and os.path.exists(PRETRAINED_ONNX):
            print(f'Loading IR-50 from {PRETRAINED_ONNX}')
            self.backbone = IR50Backbone(PRETRAINED_ONNX)
        else:
            raise FileNotFoundError(f'IR-50 not found: {PRETRAINED_ONNX}')

        self.heatmap_gen = LandmarkHeatmap(sigma=1.5)

        # CBAM with landmark prior at two stages
        self.cbam3 = CBAM(256, reduction=16, landmark_weight=landmark_weight)
        self.cbam4 = CBAM(512, reduction=16, landmark_weight=landmark_weight)

        # Multi-scale fusion
        self.proj3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fuse_weight = nn.Parameter(torch.tensor([0.3, 0.7]))

        # Self-Attention on 7×7
        self.self_attn = SelfAttentionBlock(dim=512, num_heads=8, dropout=0.1)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

        self.feat_dim = 512
        self._init_new_modules()

    def _init_new_modules(self):
        for module in [self.cbam3, self.cbam4, self.proj3,
                       self.self_attn, self.feat_bn, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                                    nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def get_param_groups(self, backbone_lr=2e-6, head_lr=5e-4):
        bb = list(self.backbone.parameters())
        hd = [p for n, p in self.named_parameters()
              if not n.startswith('backbone.')]
        return [
            {'params': bb, 'lr': backbone_lr},
            {'params': hd, 'lr': head_lr},
        ]

    def forward(self, x, landmarks=None, return_features=False):
        f3, f4 = self.backbone(x)

        # Generate landmark heatmaps at both scales (if available)
        lm_h14 = lm_h7 = None
        if landmarks is not None:
            lm_h14 = self.heatmap_gen(landmarks, 14, 14)  # (B, 1, 14, 14)
            lm_h7 = self.heatmap_gen(landmarks, 7, 7)     # (B, 1, 7, 7)

        # CBAM with landmark prior
        f3 = self.cbam3(f3, lm_h14)  # (B, 256, 14, 14)
        f4 = self.cbam4(f4, lm_h7)   # (B, 512, 7, 7)

        # Multi-scale fusion
        f3_proj = F.adaptive_avg_pool2d(self.proj3(f3), (7, 7))
        w = torch.softmax(self.fuse_weight, dim=0)
        fused = w[0] * f3_proj + w[1] * f4

        # Self-Attention
        fused = self.self_attn(fused)

        # Classify
        x = self.pool(fused).flatten(1)
        feat = self.feat_bn(x)
        logits = self.classifier(self.dropout(feat))

        if return_features:
            return logits, feat
        return logits


def build_model(num_classes=7, pretrained=True, dropout=0.4, **kwargs):
    return EmotionCNNAttention(
        num_classes=num_classes, pretrained=pretrained, dropout=dropout,
    )


if __name__ == '__main__':
    model = build_model(pretrained=True)
    x = torch.randn(2, 3, 112, 112)
    lm = torch.rand(2, 5, 2) * 112

    y1 = model(x, landmarks=lm)
    print('With landmarks:', y1.shape)

    y2 = model(x)
    print('Without landmarks:', y2.shape)

    y3, feat = model(x, landmarks=lm, return_features=True)
    print('Features:', feat.shape)

    total = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for p in model.backbone.parameters())
    print(f'Total: {total:,}  Backbone: {bb:,}  New: {total-bb:,}')
