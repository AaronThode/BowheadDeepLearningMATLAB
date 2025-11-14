#!/usr/bin/env python3
"""
Visualize the Improved Autoencoder architecture for presentations.

Outputs (to repo_root/plots):
- model_schematic_<ts>.svg/png   (clean block diagram via graphviz)
- model_graph_<ts>.svg/png       (optional detailed compute graph via torchviz)
- model_summary.txt              (optional torchinfo layer/param summary)

Dependencies (optional but recommended):
- graphviz (system): brew install graphviz
- graphviz (python): pip install graphviz
- torchviz: pip install torchviz
- torchinfo: pip install torchinfo

Usage examples:
  python visualize_architecture.py --height 121 --width 104 \
    --channels 32 64 128 --latent 64 \
    --unet-skips --upsample-conv --se-blocks --refine

Pass --no-graphviz or --no-torchviz or --no-summary to skip parts.
"""
import argparse
import os
import time
from typing import Tuple
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ImprovedAutoencoder(nn.Module):
    """Mirror of the training model with presentation toggles."""

    def __init__(
        self,
        nrow: int,
        ncol: int,
        latent_dim: int = 64,
        channels: Tuple[int, int, int] = (32, 64, 128),
        use_refine: bool = True,
        use_unet_skips: bool = False,
        use_upsample_conv: bool = False,
        use_se_blocks: bool = False,
    ):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.nrow_reduced = nrow // 8
        self.ncol_reduced = ncol // 8
        assert len(channels) == 3
        c1, c2, c3 = channels

        self.use_refine = use_refine
        self.use_skips = bool(use_unet_skips)
        self.use_upsample_conv = bool(use_upsample_conv)
        self.use_se = bool(use_se_blocks)
        self.channels = channels

        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            SEBlock(c1) if self.use_se else nn.Identity(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            SEBlock(c2) if self.use_se else nn.Identity(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            SEBlock(c3) if self.use_se else nn.Identity(),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Latent mapping
        flat = c3 * self.nrow_reduced * self.ncol_reduced
        self.to_latent = nn.Sequential(
            nn.Linear(flat, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, flat),
            nn.ReLU(inplace=True),
        )

        # Decoder
        if not self.use_upsample_conv:
            self.up1 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.up1_bn = nn.BatchNorm2d(c2)
            self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.up2_bn = nn.BatchNorm2d(c1)
            self.up3 = nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(nrow % 8, ncol % 8))
        else:
            self.up1 = nn.Conv2d(c3, c2, 3, padding=1)
            self.up1_bn = nn.BatchNorm2d(c2)
            self.up2 = nn.Conv2d(c2, c1, 3, padding=1)
            self.up2_bn = nn.BatchNorm2d(c1)
            self.up3 = nn.Conv2d(c1, 1, 3, padding=1)

        if self.use_skips:
            self.fuse1 = nn.Sequential(
                nn.Conv2d(c2 + c2, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )
            self.fuse2 = nn.Sequential(
                nn.Conv2d(c1 + c1, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse1 = None
            self.fuse2 = None

        if self.use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(1, max(8, c1 // 2), 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, c1 // 2), 1, 3, padding=1),
            )
        else:
            self.refine = None

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x); x1 = self.pool1(e1)
        e2 = self.enc2(x1); x2 = self.pool2(e2)
        e3 = self.enc3(x2); x3 = self.pool3(e3)

        flat = x3.view(x3.size(0), -1)
        z = self.to_latent(flat)
        y = self.from_latent(z)
        y = y.view(y.size(0), self.channels[2], x3.size(-2), x3.size(-1))

        if not self.use_upsample_conv:
            y = F.relu(self.up1_bn(self.up1(y)), inplace=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
            y = F.relu(self.up1_bn(self.up1(y)), inplace=True)
        if self.use_skips:
            if e2.shape[-2:] != y.shape[-2:]:
                y = F.interpolate(y, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            y = self.fuse1(torch.cat([y, e2], dim=1))

        if not self.use_upsample_conv:
            y = F.relu(self.up2_bn(self.up2(y)), inplace=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
            y = F.relu(self.up2_bn(self.up2(y)), inplace=True)
        if self.use_skips:
            if e1.shape[-2:] != y.shape[-2:]:
                y = F.interpolate(y, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            y = self.fuse2(torch.cat([y, e1], dim=1))

        if not self.use_upsample_conv:
            out = self.up3(y)
        else:
            y = F.interpolate(y, size=(self.nrow, self.ncol), mode='bilinear', align_corners=False)
            out = self.up3(y)

        if self.refine is not None:
            out = out + self.refine(out)
        return out, z


def build_schematic(args, ts_suffix: str):
    try:
        from graphviz import Digraph
    except Exception as e:
        print(f"[info] graphviz not available: {e}. Skipping schematic.")
        return

    # Ensure Graphviz 'dot' binary is available on PATH
    if shutil.which('dot') is None:
        print("[info] Graphviz 'dot' not found on PATH. Skipping schematic. Install system graphviz to enable (e.g., brew install graphviz).")
        return

    g = Digraph("Autoencoder_Schematic", format="svg")
    g.attr(rankdir="LR", concentrate="true", fontsize="10")

    # Nodes
    g.node("input", f"Input\n1x{args.height}x{args.width}", shape="box")
    g.node("enc1", f"Enc1\nConv({args.channels[0]})+BN+ReLU{'+SE' if args.se_blocks else ''}")
    g.node("pool1", "MaxPool/2")
    g.node("enc2", f"Enc2\nConv({args.channels[1]})+BN+ReLU{'+SE' if args.se_blocks else ''}")
    g.node("pool2", "MaxPool/2")
    g.node("enc3", f"Enc3\nConv({args.channels[2]})+BN+ReLU{'+SE' if args.se_blocks else ''}")
    g.node("pool3", "MaxPool/2")
    g.node("flatten", "Flatten")
    g.node("toz", f"Linear→BN→ReLU→Dropout→Linear\n(latent={args.latent})")
    g.node("fromz", "Linear→BN→ReLU→Dropout→Linear→ReLU")
    g.node("reshape", f"Reshape to {args.channels[2]}×H/8×W/8")

    up1 = "Up1: ConvT×2" if not args.upsample_conv else "Up1: Upsample×2→Conv"
    up2 = "Up2: ConvT×2" if not args.upsample_conv else "Up2: Upsample×2→Conv"
    up3 = "Up3: ConvT→1" if not args.upsample_conv else "Up3: Conv→1 (to H×W)"
    g.node("up1", up1)
    g.node("up2", up2)
    g.node("up3", up3)
    if args.refine:
        g.node("refine", "Refine: Conv→ReLU→Conv (residual)")

    g.node("output", "Output 1×H×W", shape="box")

    # Edges main path
    g.edge("input", "enc1"); g.edge("enc1", "pool1")
    g.edge("pool1", "enc2"); g.edge("enc2", "pool2")
    g.edge("pool2", "enc3"); g.edge("enc3", "pool3")
    g.edge("pool3", "flatten"); g.edge("flatten", "toz")
    g.edge("toz", "fromz"); g.edge("fromz", "reshape")
    g.edge("reshape", "up1"); g.edge("up1", "up2"); g.edge("up2", "up3")
    last = "up3"

    # Skips
    if args.unet_skips:
        g.edge("enc2", "up1", label="skip", style="dashed")
        g.edge("enc1", "up2", label="skip", style="dashed")

    if args.refine:
        g.edge("up3", "refine"); last = "refine"

    g.edge(last, "output")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.join(PLOTS_DIR, f"model_schematic_{ts_suffix}")
    try:
        g.render(base, cleanup=True)
        print(f"Saved schematic: {base}.svg")
    except Exception as e:
        print(f"[info] Failed to render schematic with graphviz: {e}. Skipping schematic.")


def build_torchviz_graph(model, height: int, width: int, ts_suffix: str):
    try:
        from torchviz import make_dot
    except Exception as e:
        print(f"[info] torchviz not available: {e}. Skipping detailed compute graph.")
        return

    x = torch.randn(1, 1, height, width)
    y, z = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    base = os.path.join(PLOTS_DIR, f"model_graph_{ts_suffix}")
    try:
        dot.format = 'svg'
        dot.render(base, cleanup=True)
        print(f"Saved detailed compute graph: {base}.svg")
    except Exception as e:
        print(f"[info] Failed to render SVG, trying PNG: {e}")
        try:
            dot.format = 'png'
            dot.render(base, cleanup=True)
            print(f"Saved detailed compute graph: {base}.png")
        except Exception as e2:
            print(f"[info] Failed to render PNG as well: {e2}. Skipping detailed compute graph.")


def write_torchinfo_summary(model, height: int, width: int):
    try:
        from torchinfo import summary
    except Exception as e:
        print(f"[info] torchinfo not available: {e}. Skipping summary.")
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    info = summary(model, input_size=(1, 1, height, width), col_names=("input_size", "output_size", "num_params", "kernel_size"))
    out_path = os.path.join(PLOTS_DIR, "model_summary.txt")
    with open(out_path, "w") as f:
        f.write(str(info))
    print(f"Saved torchinfo summary: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize autoencoder architecture")
    parser.add_argument("--height", type=int, default=121)
    parser.add_argument("--width", type=int, default=104)
    parser.add_argument("--channels", type=int, nargs=3, default=[32, 64, 128])
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--unet-skips", dest="unet_skips", action="store_true")
    parser.add_argument("--upsample-conv", dest="upsample_conv", action="store_true")
    parser.add_argument("--se-blocks", dest="se_blocks", action="store_true")
    parser.add_argument("--no-graphviz", action="store_true", help="Skip block diagram")
    parser.add_argument("--no-torchviz", action="store_true", help="Skip detailed compute graph")
    parser.add_argument("--no-summary", action="store_true", help="Skip torchinfo summary")
    args = parser.parse_args()

    ts_suffix = time.strftime("%Y%m%d-%H%M%S")

    model = ImprovedAutoencoder(
        nrow=args.height,
        ncol=args.width,
        latent_dim=args.latent,
        channels=tuple(args.channels),
        use_refine=args.refine,
        use_unet_skips=args.unet_skips,
        use_upsample_conv=args.upsample_conv,
        use_se_blocks=args.se_blocks,
    )
    model.eval()

    if not args.no_graphviz:
        build_schematic(args, ts_suffix)
    if not args.no_torchviz:
        build_torchviz_graph(model, args.height, args.width, ts_suffix)
    if not args.no_summary:
        write_torchinfo_summary(model, args.height, args.width)


if __name__ == "__main__":
    main()
