import glob
import numpy as np
from imageio import imread
from collections import Counter

mask_globs = [
    '/workspace/mmsegmentation_xview2/dataset/masks/*.png',
    '/workspace/mmsegmentation_xview2/dataset/masks_val/*.png',
    '/workspace/mmsegmentation_xview2/dataset/holdout/*.png',
    '/workspace/mmsegmentation_xview2/dataset/holdout/masks_test/*.png',
]
paths = []
for g in mask_globs:
    paths.extend(sorted(glob.glob(g)))

if not paths:
    print("No mask files found. Check the globs above and your dataset layout.")
    raise SystemExit(1)

total_counts = Counter()
per_image_pos = []
example_summary = []

for p in paths:
    img = imread(p)
    # convert RGB/paletted to single channel if needed
    if img.ndim == 3:
        img = img[..., 0]
    vals, counts = np.unique(img, return_counts=True)
    for v, c in zip(vals, counts):
        total_counts[int(v)] += int(c)
    # define positive candidate as value==1 if present, else try to infer later
    pos_count = int((img == 1).sum()) if 1 in vals else 0
    per_image_pos.append(pos_count / img.size)
    example_summary.append((p, dict(zip(vals.tolist(), counts.tolist())), per_image_pos[-1]))

total_pixels = sum(total_counts.values())
print("Unique label ids across dataset and their pixel counts:")
for v, c in sorted(total_counts.items()):
    print(f"  {v}: {c} ({c/total_pixels:.4f} fraction)")

if 1 in total_counts:
    pos_frac = total_counts[1] / total_pixels
    print(f"\nOverall fraction of pixels with value==1: {pos_frac:.6f}")
else:
    print("\nNo pixels with value==1 found in dataset.")

if 0 in total_counts:
    zero_frac = total_counts[0] / total_pixels
    print(f"Overall fraction of pixels with value==0: {zero_frac:.6f}")

if 255 in total_counts:
    print("Found 255 (often used as ignore_index).")

# per-image stats summary
per_image_pos = np.array(per_image_pos)
print(f"\nPer-image positive fraction (value==1) mean={per_image_pos.mean():.6f} std={per_image_pos.std():.6f}")
print("Examples (path, unique_value_counts, pos_fraction):")
for p, vc, pf in example_summary[:10]:
    print(f"  {p}: {vc} pos_frac={pf:.6f}")

# quick verdict heuristic:
# if positive fraction ~0.03 -> building likely encoded as 1 (minority)
# if positive fraction ~0.97 -> building likely encoded as 0 (inverted)
if 1 in total_counts:
    if total_counts[1] / total_pixels < 0.2:
        print("\nHeuristic: value==1 appears to be the MINORITY class (likely building).")
    elif total_counts[1] / total_pixels > 0.8:
        print("\nHeuristic: value==1 appears to be the MAJORITY class (labels may be inverted).")
else:
    # if no 1s, check whether 0 is minority
    if 0 in total_counts and total_counts[0] / total_pixels < 0.2:
        print("\nHeuristic: value==0 appears to be the MINORITY class (labels may be inverted).")
    else:
        print("\nHeuristic: could not determine positive label (no value 1 found). Inspect sample masks above.")