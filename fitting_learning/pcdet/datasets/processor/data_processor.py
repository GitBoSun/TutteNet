from functools import partial
import numba as nb

import numpy as np
import torch
from skimage import transform
from sklearn.neighbors import NearestNeighbors as NN


from ...utils import common_utils
from collections import defaultdict

tv = None
try:
    import cumm.tensorview as tv
except:
    pass
class DataProcessor(object):
    def __init__(self, processor_configs,  training, ):
        self.training = training
        self.mode = 'train' if training else 'test'
        self.data_processor_queue = []


        self.num_extra_point_features = 0
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            if cur_cfg.NAME == "attach_spherical_feature":
                self.num_extra_point_features += 3
            self.data_processor_queue.append(cur_processor)

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['point_wise']['point_xyz']
            shuffle_idx = np.random.permutation(points.shape[0])
            data_dict['point_wise'] = common_utils.filter_dict(
                                          data_dict['point_wise'],
                                          shuffle_idx
                                      )

        return data_dict
    
    def limit_num_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.limit_num_points, config=config)

        max_num_points = config["MAX_NUM_POINTS"]

        points = data_dict['point_wise']['point_xyz']
        if points.shape[0] > max_num_points and self.training:
            subsampler = config.get("SUBSAMPLE_METHOD", 'UNIFORM')
            if subsampler == 'UNIFORM':
                shuffle_idx = np.random.permutation(points.shape[0])[:max_num_points]
                data_dict['point_wise'] = common_utils.filter_dict(
                                              data_dict['point_wise'],
                                              shuffle_idx
                                          )
            elif subsampler == 'FPS':
                points = torch.from_numpy(points)
                ratio = max_num_points / points.shape[0]
                fps_index = fps(points, ratio=ratio)
                data_dict['point_wise'] = common_utils.filter_dict(
                                              data_dict['point_wise'],
                                              fps_index
                                          )
            elif subsampler == 'GRID':
                points = torch.from_numpy(points)
                grid_size = torch.tensor(config["GRID_SIZE"]).float()
                cluster = grid_cluster(points, grid_size)
                unique, inv = torch.unique(cluster, return_inverse=True)
                subindices = scatter(torch.arange(points.shape[0]), inv, dim=0, dim_size=unique.shape[0], reduce='max')
                data_dict['point_wise'] = common_utils.filter_dict(data_dict['point_wise'], subindices.numpy())

        return data_dict

 
    def random_generate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_generate, config=config)
        self.a_range = config["A_RANGE"]
        self.b_range = config["B_RANGE"]
        self.num_points = config["NUM_POINTS"]
        a = np.random.rand() * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        b = np.random.rand() * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        theta = 2*np.pi*np.arange(self.num_points).astype(np.float32)/self.num_points
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        init_points = np.concatenate((np.expand_dims(x, 1),  np.expand_dims(y, 1)), axis=1)

        alpha = np.random.rand() * np.pi * 2
        rot_matrix = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        rot_points = np.matmul(init_points, rot_matrix)

        trans_limit = (1 - rot_points.max(0)) * 0.9
        trans_xy = (np.random.rand(2)-0.5)*2 * trans_limit
        trans_points = rot_points + trans_xy
        
        data_dict['point_wise']['points'] = trans_points 
        data_dict['point_wise']['e_param']=np.array([a,b, alpha,trans_xy[0], trans_xy[1]])
        # print('??', np.array([a,b, alpha,trans_xy[0], trans_xy[1]]))        
        return data_dict


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
