# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import pdb
import cv2
import numpy as np
from PIL import Image

@DATASETS.register_module()
class custom3DDataset_28(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('other', 'Cork Natural', 'Fabric Lace', 'Clay Terracotta', 'Porcelain001', 'Metal Galvanized',
                 'Metal Sandblasted', 'Concrete Asphalt', 'Rope001', 'Paper005',
                 'Wood Rough', 'Plastic013A', 'Leather Skin', 'Baked Lighting Material', 'Metal Foil', 'Clay Earthenware', 'Concrete Coarse',
                 'Fabric Linen', 'Fabric Felte', 'Bricks090', 'rustMetal022', 'Facade001', 'Carbon Fiber', 'Metal Brushed', 'Paint Roll', 'Rubber001', 
                 'Plastic Composite', 'Wood Slice', 'Fabric Nylon'),
        palette=[[0, 0, 0], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [100, 100, 0], [128, 64, 128], [255, 255, 255], [100, 0, 0],
                 [0, 255, 0], [0, 100, 0], [0, 255, 255], [0, 140, 140], [255, 0, 255], [100, 0, 100]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_222_new.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs)

    def get_cat_ids(self, idx):
        # pdb.set_trace()
        data_info = self.get_data_info(idx)
        label = np.array(Image.open(data_info['seg_map_path']))
        label_list = np.unique(label).tolist()
        return [int(i) for i in label_list]

