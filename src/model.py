import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DenseUNet(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.densenet121(weights=weights).features

        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0)
        self.pool0 = f.pool0
        self.block1 = f.denseblock1
        self.trans1 = f.transition1
        self.block2 = f.denseblock2
        self.trans2 = f.transition2
        self.block3 = f.denseblock3
        self.trans3 = f.transition3
        self.block4 = nn.Sequential(f.denseblock4, f.norm5, nn.ReLU(inplace=True))

        self.up4 = UpBlock(1024, 1024, 512, dropout)
        self.up3 = UpBlock(512, 512, 256, dropout)
        self.up2 = UpBlock(256, 256, 128, dropout)
        self.up1 = UpBlock(128, 64, 64, dropout)
        self.up0 = ConvBlock(64, 32, dropout)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.block1(self.pool0(s0))
        s2 = self.block2(self.trans1(s1))
        s3 = self.block3(self.trans2(s2))
        b = self.block4(self.trans3(s3))

        d = self.up4(b, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d = self.up0(d)
        return self.head(d)


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseUNet(pretrained=True).to(dev)
    enc = sum(p.numel() for n, p in model.named_parameters() if n.split(".")[0] in {"stem", "block1", "trans1", "block2", "trans2", "block3", "trans3", "block4"})
    tot = sum(p.numel() for p in model.parameters())
    print(f"params: total {tot/1e6:.1f}M | encoder (DenseNet-121) {enc/1e6:.1f}M | decoder {(tot-enc)/1e6:.1f}M")

    x = torch.randn(2, 3, 512, 512, device=dev)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print("input ", tuple(x.shape))
    print("output", tuple(logits.shape), "-> probability map via sigmoid, range",
          f"[{torch.sigmoid(logits).min():.3f}, {torch.sigmoid(logits).max():.3f}]")

    enable_mc_dropout(model)
    with torch.no_grad():
        p1, p2 = torch.sigmoid(model(x)), torch.sigmoid(model(x))
    print(f"MC-dropout: two stochastic passes differ by mean |dp| = {(p1-p2).abs().mean():.4f} (>0 means dropout is active)")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model.train()
        out = model(x); out.mean().backward()
        print(f"train step, batch 2 @512x512: peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
