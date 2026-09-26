import numpy as np
import cv2


class TrainAugment:
    def __init__(self, max_rotate=10.0, max_zoom=0.10, brightness=0.15, contrast=0.15, p=0.8, seed=None):
        self.max_rotate = max_rotate
        self.max_zoom = max_zoom
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, img, msk):
        if self.rng.random() < self.p:
            img, msk = self._affine(img, msk)
        if self.rng.random() < self.p:
            img = self._intensity(img)
        return img, msk

    def _affine(self, img, msk):
        h, w = img.shape
        angle = self.rng.uniform(-self.max_rotate, self.max_rotate)
        scale = 1.0 + self.rng.uniform(-self.max_zoom, self.max_zoom)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, msk

    def _intensity(self, img):
        b = self.rng.uniform(-self.brightness, self.brightness)
        c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        img = (img - 0.5) * c + 0.5 + b
        return np.clip(img, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from dataset import load_frame, PlacentaDataset

    frame = load_frame(("ssh_TSE",), "train")
    base = PlacentaDataset(frame, transform=None)
    aug = PlacentaDataset(frame, transform=TrainAugment(seed=0))

    idx = [0, 400, 900, 1500]
    fig, ax = plt.subplots(2, len(idx) * 2, figsize=(4 * len(idx) * 2, 8))
    for k, i in enumerate(idx):
        (im0, mk0, meta), (im1, mk1, _) = base[i], aug[i]
        for j, (im, mk, title) in enumerate([(im0, mk0, "original"), (im1, mk1, "augmented")]):
            col = k * 2 + j
            a, m = im[0].numpy(), mk[0].numpy()
            ax[0, col].imshow(a, cmap="gray", vmin=0, vmax=1)
            ax[0, col].set_title(f"{meta['patient']} s{meta['slice']} {title}", fontsize=10)
            ax[1, col].imshow(a, cmap="gray", vmin=0, vmax=1)
            ax[1, col].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45)
            ax[1, col].set_title(f"mask px {int(m.sum())}  unique {np.unique(m).tolist()}", fontsize=9)
            ax[0, col].axis("off"); ax[1, col].axis("off")
    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / "data" / "_augment_check.png"
    plt.savefig(out, dpi=60)
    print("saved", out)
