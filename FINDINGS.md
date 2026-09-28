# Does the model know when it is wrong?

Findings for RQ1 on the Mendeley placenta accreta MRI dataset.

## Setup

The Mendeley dataset (284gwmf5bh) has 131 patients, all with PAS, and comes with a placenta outline for every slice but no diagnosis labels. So I could not train the yes/no classifier from the Zhang paper. Instead I trained a DenseNet-121 U-Net to outline the placenta and asked the same question at the pixel level: when the outline is wrong, does the model's confidence show it?

Data: ssh_TSE sequence, 3695 slices. Split by patient, 93 train / 20 val / 20 test. The test patients were not touched until the model was final.

Model: DenseNet-121 encoder (ImageNet weights) with a U-Net decoder, dropout 0.2 in the decoder. Trained 30 epochs, kept the best validation epoch (12). Test Dice 0.865 mean, 0.893 median. The Sheffield group got 0.885 with a plain U-Net on the same data, so this is in the normal range.

Confidence is max(p, 1-p) per pixel. Uncertainty is the std over 20 forward passes with dropout left on (MC dropout).

## What the plots show

### Reliability diagram

Over all pixels the model looks very well calibrated, ECE 0.011. But 97% of pixels are background and the model is trivially sure about those. If you only look at pixels within 5 px of the true placenta edge, ECE jumps to 0.197. When the model says it is 99% sure about a boundary pixel it is right 83% of the time. In the 0.90 to 0.95 confidence bin it is right about 67% of the time.

Every bin below the top one sits 15 to 25 points under the diagonal in all three views. The gap is hidden by the background, not absent.

### Confidence histograms

Of the 32,317 wrong pixels in the test set, 32% were made with confidence above 0.99. The median confidence on a wrong pixel is 0.972, on a right pixel 0.9998. They overlap a lot.

At the slice level, comparing good slices (Dice at least 0.8, n=470) to poor ones (n=88): whole-slice confidence is 0.9966 vs 0.9963. No difference. Confidence at the true edge is actually higher on poor slices, 0.948 vs 0.933. MC dropout std on the placenta is higher on poor slices, 0.020 vs 0.015, but the two groups overlap too much to set a threshold.

### High confidence errors

The six worst slices (Dice under 0.7, ranked by number of wrong pixels above 0.95 confidence) all look the same. The model draws a clean, sharp outline around part of the placenta and cuts the rest off. The missed part gets probability near zero and MC std near zero. It is not a little unsure about the missed region. It has no doubt at all. Up to 16,500 wrong pixels per slice at confidence above 0.95, with whole-slice confidence 0.995 to 0.998 on every one.

Three of the six are the first slices of patient sub050's stack. Two are mid-stack slices of sub099 where the placenta is long and thin.

### Low confidence correct

The six most hesitant slices among the good ones (Dice at least 0.9) show uncertainty only as a thin ring along the predicted edge, at the tapered tips and where the placenta touches the fetus. Never in the interior. The most hesitant correct slice in the whole test set has about 1,500 uncertain pixels. The worst error slice has 16,500 confident wrong pixels. The model hesitates about ten times less on its right answers than it is certain on its wrong ones.

## Patterns

Checked across all 558 test slices.

Held up:
- Edge of volume. First and last slices of a stack are poor 35% of the time, interior slices 10%. Dice 0.79 vs 0.88.
- Small placenta. Slices where the placenta is at its thinnest for that patient are the worst (relative area, Spearman 0.30 with Dice).
- The boundary is where everything happens. Pixels within 5 px of the true edge are 3% of the image but 54% of the errors. At 0 to 1 px from the edge accuracy is 56% and confidence is 91%.
- Errors bunch up in a few patients. The worst 3 of 20 patients produce 29% of all confident wrong pixels. 8 of 20 have no poor slices at all.
- Confidence at the true boundary is the best single predictor of a bad slice, Spearman -0.60, but inverted. If the model is very sure right where the true edge is, its own edge is somewhere else.
- MC dropout std has a weak relation to Dice, -0.37. Some signal, not enough to filter on.

Did not hold up:
- Scanner cohort. The `sub` and `v` groups had the same Dice (0.865 vs 0.866) and the same uncertainty. Only 3 `v` patients in the test set though.
- Shape. Elongation and number of separate pieces did not matter.
- Overall bias. I first thought the model under-segments because the worst slices are all misses. Across the whole test set FN/FP is 1.05 and predicted area over true area is 0.99. It is unbiased on average. The truncation pattern is only how the biggest failures look.

## Domain shift

I ran the same model on the BTFE sequence of the same 20 test patients. It had never seen BTFE.

Dice fell from 0.865 to 0.819. Poor slice rate doubled, 16% to 30%. Slices under Dice 0.5 went from 5 to 15. Dice fell in 16 of 19 patients, by up to 0.16.

Whole-slice confidence went from 0.9964 to 0.9956. It did not move. Boundary ECE went 0.197 to 0.226, placenta pixel ECE 0.100 to 0.178. Accuracy at 99% confidence on the boundary fell from 83% to 78%.

MC dropout std went from 0.017 to 0.024, p=7e-25. That was the only number that reacted to the shift.

A new kind of error showed up. On ssh_TSE the model mostly missed parts of the placenta. On BTFE, where fluid is bright, it also outlines things that are not placenta. sub066 slice 7 has 23,000 false positive pixels above 0.95 confidence and zero missed placenta. In sub113 slice 22 it fires on the fetal body. The `v` cohort, which was fine on ssh_TSE, dropped to 0.771.

## Answer to RQ1

Does the model know when it might be wrong? Mostly no.

Its confidence as normally reported (softmax, averaged over the image) says 99.6% on every slice, good or bad, on the training sequence or a new one. That matches what Zhang et al. reported for their classifier: high probability scores even when wrong. Here it happens because the background swamps the average, and because the largest errors are clean cuts with no doubt attached.

Where the model does hesitate, the hesitation is in the right place, at the real edges and tips. It is just far too small compared to the size of the confident errors.

MC dropout is the only thing that carried any information. It separates good from poor slices weakly, and it was the only signal that rose when the input distribution changed. It would not catch a truncation, but it would flag a distribution shift.

Does confidence behave differently on difficult cases? The difficult cases (edge slices, thin placenta, new sequence) are exactly where the model is most confidently wrong. Confidence goes the wrong way.

## Limits

This is segmentation, not diagnosis. Nothing here says anything about a missed PAS case or a false alarm on a healthy woman. That needs labels the dataset does not have.

20 test patients. Three of them are the `v` cohort. Some patterns are probably real but underpowered.

One model, one training run, one seed. The numbers would move a little with another seed.

Whole-slice confidence is a poor summary for segmentation and I used it mainly to show that. The boundary and placenta-only views are the honest ones.

Ground truth is one radiologist's outline. Some of the "errors" at the edge are probably disagreements a second radiologist would also have.

## Files

- `src/predict.py` produces slices.csv, pixels.parquet, maps.npz per run
- `src/analysis/calibration.py` reliability diagram and ECE
- `src/analysis/histograms.py` confidence histograms
- `src/analysis/gallery.py` the two example galleries
- `src/analysis/patterns.py` pattern statistics
- Outputs in `runs/full_ssh_tse/pred_test_ssh_TSE/` and `pred_test_BTFE/` (not in git)
