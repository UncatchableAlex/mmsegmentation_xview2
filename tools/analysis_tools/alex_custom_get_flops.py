from mmengine.config import Config
from mmseg.apis import init_model
from mmcv.cnn import get_model_complexity_info
import torch

# Load the exact same config
#cfg = Config.fromfile('configs/convnext/convnext-tiny_upernet_xview2.py') #output: FLOPs: 233.44 GFLOPs, Params: 60.13 M
cfg = Config.fromfile('configs/convnext/convnext-tiny-ablation_upernet_xview2.py') # output: FLOPs: 910.58 GFLOPs, Params: 60.14 M
# Build model exactly as train.py would
model = init_model(cfg, device='cuda' if torch.cuda.is_available() else 'cpu')

# Switch to eval mode for deterministic shapes
model.eval()

# Measure FLOPs and params for 512×512 input
flops, params = get_model_complexity_info(
    model,
    (3, 512, 512),
    as_strings=True,
    print_per_layer_stat=True,  # <- gives per-layer breakdown
    #verbose=True
)

print(f'FLOPs: {flops}, Params: {params}')