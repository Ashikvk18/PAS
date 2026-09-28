# PAS Project — Progress Log

Evaluating Confidence and Uncertainty in MRI-Based Deep Learning Models for Placenta Accreta Spectrum.
Every step gets a status: ✅ done · ❌ failed · ⏳ in progress · ⬜ not started · ⚠️ done with caveat

---

## Phase 0 — Understanding & literature

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0.1 | Read project brief (Details.docx) | ✅ | RQ1 = confidence/uncertainty; RQ2/RQ3 TBD |
| 0.2 | Read Zhang 2025 (DenseNet-121) | ✅ | Full text (PMC). Ext. AUC 0.863, ext. sensitivity fell to 62.5% |
| 0.3 | Read Wang 2024 (nnU-Net + DenseNet) | ⚠️ | Paywalled. Read full PubMed abstract + ISMRM 2023 conference version |
| 0.4 | Read Zheng 2024 (DRC model) | ⚠️ | Paywalled. Read full PubMed abstract + 2025 review summary |
| 0.5 | Read professor's files (Notes, Manuscript, PAI 2014, GitHub skeleton) | ✅ | Lab favours ResNet/DenseNet; U-Net mentioned for segmentation |
| 0.6 | Read Dr. Sumi's files (BCM reading list, Images.pptx) | ✅ | Radiomics-oriented lit base; slides = reference PAS signs |
| 0.7 | Search for released code / trained models from the 3 papers | ❌ | None released. Found unrelated repos (PASD, HACL-Net, nnU-Net) |

## Phase 1 — Dataset (Mendeley 284gwmf5bh)

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1.1 | Download dataset | ✅ | 172 MB zip → 4 RAR archives |
| 1.2 | Extract archives | ⚠️ | 3/4 clean. BTFE_image.rar (re-downloaded Sep 17) was corrupt → restored original from zip. Corrupt copy kept as `BTFE_image_CORRUPT_sep17.rar` |
| 1.3 | Inventory | ✅ | ssh_TSE: 3,695 img/mask, 132 pts. BTFE: 3,411 img/mask, 131 pts. 512×512 JPEG; masks binary PNG, **variable size** (must resize) |
| 1.4 | Find PAS / non-PAS labels | ❌ | **None exist.** Sheffield BiGC U-Net paper (IEEE JBHI 2025) confirms: all 131 pts are PAS, "metadata and subtypes not available" |
| 1.5 | Ask Dr. Sumi for labels (via Farhan) | ⏳ | Message sent Sep 20. Waiting |
| 1.6 | **Decision: use Mendeley only → segmentation + uncertainty** | ✅ | Classification impossible without labels. DenseNet-121 used as U-Net encoder. Switch to classifier if labels arrive |

## Phase 2 — Environment

| # | Step | Status | Notes |
|---|------|--------|-------|
| 2.1 | Check Python / packages / GPU | ✅ | Python 3.13.7 · numpy, pandas, matplotlib, PIL, sklearn, scipy, cv2, tqdm present · **no TF / PyTorch yet** · GPU: RTX 5070 Laptop 8 GB · 16 cores · 604 GB free |
| 2.2 | Choose framework | ✅ | **PyTorch.** TF has no native Windows GPU support; RTX 5070 (Blackwell) needs CUDA 12.8+ which PyTorch ≥2.7 ships with |
| 2.3 | Install framework + verify GPU visible | ✅ | torch 2.11.0+cu128, torchvision 0.26.0. CUDA visible, compute cap 12.0, 8.5 GB VRAM. DenseNet-121 forward pass on 512×512 OK (8.0M params) |
| 2.4 | Install 7-Zip (for RAR extraction) | ✅ | via winget |

## Phase 3 — Data pipeline

| # | Step | Status | Notes |
|---|------|--------|-------|
| 3.1 | Build patient index (patient → slices, cohort `sub`/`v`, sequence) | ✅ | `src/build_index.py` → `data/index.csv`. 7,106 rows, every image has a mask. **130 patients appear in both sequences** → split must be by patient across sequences |
| 3.2 | Patient-level train / val / test split | ✅ | `src/make_split.py` → `data/split.csv`. 93 / 20 / 20 patients (seed 42, stratified by cohort; `v` = 12/2/3). Same split applied to both sequences. Leakage check passed |
| 3.3 | Dataset loader: image + mask, resize masks to 512×512 | ✅ | `src/dataset.py`. Image → [0,1], repeated to 3 ch (for ImageNet weights); mask → nearest-resize → binary. Batch (8,3,512,512) / (8,1,512,512). ssh_TSE train/val/test = 2,582 / 555 / 558 slices |
| 3.4 | Augmentation (flip / rotate / brightness, per professor TODO) | ✅ | `src/augment.py`. Rotation ±10°, zoom ±10% (same affine on image+mask, nearest for mask), brightness ±0.15, contrast ±15%. **No flips** — sagittal anatomy isn't symmetric. Verified in `data/_augment_check.png`, masks stay binary |
| 3.5 | Visual sanity check of a batch | ✅ | `data/_batch_check.png` — masks align with placenta on all 8 sampled slices incl. a `v` case |

## Phase 4 — Model

| # | Step | Status | Notes |
|---|------|--------|-------|
| 4.1 | Understand DenseNet-121 architecture (dense blocks, transitions) | ✅ | Walked through: 4 dense blocks (6/12/24/16 layers), growth rate 32, concatenation not replacement, transitions do all downsampling. Skip taps at 1/2, 1/4, 1/8, 1/16, 1/32 |
| 4.2 | Build DenseNet-121-encoder U-Net | ✅ | `src/model.py` `DenseUNet`. ImageNet-pretrained encoder (7.0M) + decoder (15.7M) = 22.6M. In (2,3,512,512) → out (2,1,512,512) logits. Train step batch 2 = 2.5 GB VRAM → batch 6 should fit in 8 GB |
| 4.3 | Add dropout for MC-Dropout uncertainty | ✅ | `Dropout2d(0.2)` in every decoder block; `enable_mc_dropout()` keeps them stochastic at inference. Verified two passes differ |
| 4.4 | Short smoke-test training (1–2 epochs) | ✅ | `src/train.py` (BCE+Dice, AdamW, encoder LR ×0.1, cosine, bf16 autocast). 2 epochs, batch 6, ~3.4 min/epoch. **Val Dice 0.817 → 0.838.** Visual check `runs/smoke/_val_preds.png`: 5/6 good; `v027 s41` fails (Dice 0.25) with visibly low-confidence output → early preview of RQ1 |
| 4.5 | Full training on ssh_TSE | ✅ | Run `full_ssh_tse`, 30 epochs, ~4 min/epoch. **Best val Dice 0.871 @ epoch 12** → `checkpoints/full_ssh_tse_best.pt`. Val plateaued ~0.87 from epoch 12 while train reached 0.954 (mild overfit; best-epoch checkpoint kept). Power settings restored to original (sleep 5 min, display 5 min) |
| 4.6 | Evaluate Dice on held-out test patients | ✅ | `src/predict.py` → `runs/full_ssh_tse/pred_test_ssh_TSE/` (slices.csv, pixels.parquet, maps.npz). 558 slices / 20 pts. **Test Dice: mean 0.865, median 0.893**, patient-level 0.865. `sub` 0.865 vs `v` 0.866 — no cohort gap. 5 slices Dice<0.5, 38 <0.7. Worst patients: sub050 (0.72), sub090 (0.75), sub099 (0.78), v006 (0.79) |

## Phase 5 — Uncertainty analysis (RQ1)

| # | Step | Status | Notes |
|---|------|--------|-------|
| 5.1 | Per-pixel confidence maps + MC-Dropout variance | ✅ | 20 MC-dropout passes per slice; 4,000 sampled pixels/slice (2.23 M) with prob, correct, mc_std, boundary distance. **Early findings:** whole-slice mean confidence ≈ 0.996 even on Dice-0 slices (background dominates → uninformative, the segmentation analogue of Zhang's overconfidence). MC-std lights up only at the *edges of what was predicted*; missed placenta regions show ~zero uncertainty = confident false negatives. Boundary-band confidence correlates −0.63 with Dice |
| 5.2 | Reliability diagram (pixel confidence vs accuracy) + ECE | ⬜ | |
| 5.3 | Confidence histograms (correct vs incorrect) | ⬜ | |
| 5.4 | Gallery: high-confidence errors | ⬜ | |
| 5.5 | Gallery: low-confidence correct | ⬜ | |
| 5.6 | Pattern analysis (placenta size, slice position, `sub` vs `v`, boundary vs interior) | ⬜ | |
| 5.7 | Domain-shift test: run ssh_TSE model on BTFE | ⬜ | |
| 5.8 | Write-up of findings | ⬜ | |

---

## Open questions / blockers
- Labels from Dr. Sumi — pending (1.5). If they arrive: add DenseNet classifier head, reuse encoder.
- Professor's skeleton expects `Healthy/` + `Placenta_Accreta/`; we will use `images/` + `masks/`. Tell her.
- Wang & Zheng full texts — need library access or professor's PDFs.

## Housekeeping to decide
- Move `BCM ML + PAI Studies .docx`, `Images.pptx` into `PAS Project/`?
- Delete `BTFE_image_CORRUPT_sep17.rar` (118 MB) and stray `JPEGImages/` (12 files)?
- Keep or delete `_inspect_masks.png`?
