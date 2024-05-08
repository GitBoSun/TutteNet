from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pickle
import torch.utils.data as torch_data

from ..utils import common_utils
#from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        if self.dataset_cfg is None:
            return

        self.training = training

        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

        self.num_point_features = dataset_cfg.get("NUM_POINT_FEATURES", 0)

        # self.data_augmentor = DataAugmentor(
        #     self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.box_classes, logger=self.logger
        # ) if self.training else None
        # self.data_processor = DataProcessor(
        #     self.dataset_cfg.DATA_PROCESSOR, 
        #     training=self.training, 
        # )
        self.data_augmentor = None
        self.data_processor = None

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False


        self.depth_downsample_factor = None

        self.runtime_cfg = dict(
            num_point_features=self.num_point_features,
        )
        if self.logger is not None:
            self.logger.info(f"{self.__class__} Dataset in {self.mode} mode.")

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        print(f'getting state {d}')
        del d['logger']
        return d

    def __setstate__(self, d):
        print(f'setting state {d}')
        self.__dict__.update(d)


    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.
        Args:
            index:
        Returns:
        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """

        if self.training and self.data_augmentor is not None:
            data_dict = self.data_augmentor.forward(data_dict)

        if self.data_processor is not None:
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False, num_mix3d_samples=1):

        data_dict = defaultdict(lambda: defaultdict(list))
        for cur_sample in batch_list:
            for key0, val0 in cur_sample.items():
                for key, val in val0.items():
                    data_dict[key0][key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key0, val0 in data_dict.items():
            for key, val in val0.items():
                try:
                    if not isinstance(val, list):
                        ret[key] = val
                    elif key in [ 'spv_instance_label_back', 'strat_point_feat',
                              ]:
                        ret[key] = np.concatenate(val, axis=0)
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

        ret['batch_size'] = batch_size
        return ret
