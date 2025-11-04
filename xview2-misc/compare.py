import os
import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm
from mmseg.apis import init_model, inference_model

# -----------------------------
# User settings
# -----------------------------

# Configs & checkpoints for each model:
CONVNEXT_CFG = 'configs/convnext/convnext-tiny_upernet_xview2.py'
CONVNEXT_CKPT = 'work_dirs/baseline5/best_mDice_iter_40000.pth'

RESNET_CFG = 'configs/convnext/convnext-tiny-ablation_upernet_xview2.py'
RESNET_CKPT = 'work_dirs/ablation1/best_mDice_iter_38000.pth'

# Dataset directories:
IMG_DIR  = 'dataset/holdout/images_test'
MASK_DIR = 'dataset/holdout/masks_test'

DEVICE = 'cuda:0'
OUTPUT_CSV = 'stem_iou_differences.csv'

# -----------------------------
# IoU computation
# -----------------------------
def compute_single_iou(pred_mask, gt_mask):
    """Compute IoU for class 1 (building)."""
    intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()
    return intersection / union if union != 0 else 0.0

def infer_mask(model, img_path):
    """Run MMSegmentation inference and return numpy mask."""
    result = inference_model(model, img_path)
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
    return pred_mask.astype(np.int64)

# -----------------------------
# Main comparison
# -----------------------------
def main():
    print('Loading models...')

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    results = np.zeros((2, len(image_files), 2))

    print('Comparing IoUs...')
    for (i, (cfg, ckpt)) in enumerate([(CONVNEXT_CFG, CONVNEXT_CKPT), (RESNET_CFG, RESNET_CKPT)]):
        model = init_model(cfg, ckpt, device=DEVICE)
        for (j,img_name) in tqdm(enumerate(image_files)):
            img_path = os.path.join(IMG_DIR, img_name)
            gt_path  = os.path.join(MASK_DIR, img_name)

            if not os.path.exists(gt_path):
                print(f"path not found: {gt_path}")

            gt_mask = mmcv.imread(gt_path, flag='unchanged').astype(np.int64)
            pixels = np.sum(gt_mask)
            pred = infer_mask(model, img_path)
            iou = compute_single_iou(pred, gt_mask)
            results[i][j][0] = iou
            results[i][j][1] = pixels
            

            # results.append({
            #     'image': img_name,
            #     'IoU_ConvNeXtStem': iou_convnext,
            #     'IoU_ResNetStem': iou_resnet,
            #     'AbsDiff': abs_diff
            # })

    # df = pd.DataFrame(results)
    # df = df.sort_values(by='AbsDiff', ascending=False)
    # df.to_csv(OUTPUT_CSV, index=False)

    # print(f'\nSaved IoU differences to {OUTPUT_CSV}')
    # print(f'Largest difference: {df.iloc[0].AbsDiff:.4f} on {df.iloc[0].image}')
    #diffs = results[0] - results[1]
    #abs_diffs = np.abs(diffs)
    #print(f'max difference: {np.max(abs_diffs):.4f} on file {image_files[np.argmax(abs_diffs)]}')

    pd_res = []
    for i, img_name in enumerate(image_files):
        score_convnext = results[0][i][0] * results[0][i][1]
        score_resnet  = results[1][i][0] * results[1][i][1]

        pd_res.append({
            'image': img_name,
            'IoU_ConvNeXt': results[0][i][0],
            'IoU_ResNet': results[1][i][0],
            'pixels': results[0][i][1],
            'score_convnext': results[0][i][0] * results[0][i][1],
            'score_resnet': results[1][i][0] * results[1][i][1],
            'score_diff' : np.abs(score_convnext - score_resnet),
            #'AbsDiff': abs_diffs[i],
        })
    df = pd.DataFrame(pd_res)
    print(df.sort_values(by='score_diff', ascending=False).head(10))
    df.to_csv(OUTPUT_CSV, index=False)
if __name__ == '__main__':
    main()
