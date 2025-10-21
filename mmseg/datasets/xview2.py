from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class CustomDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0,0,0], [255, 255, 0]],
      #  label_map={1: 0, 0: 255}
    )    
    
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)