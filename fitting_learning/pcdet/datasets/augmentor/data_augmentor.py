from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler, semantic_sampler, semantic_seg_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def set_epoch(self, epoch):
        self.epoch = epoch
        for aug in self.data_augmentor_queue:
            if hasattr(aug, 'set_epoch'):
                aug.set_epoch(epoch)

    #def semantic_sampling(self, config=None):
    #    seg_sampler = semantic_sampler.SemanticSampler(
    #        root_path=self.root_path,
    #        sampler_cfg=config,
    #        class_names=self.class_names,
    #        logger=self.logger
    #    )
    #    return seg_sampler
    
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def semantic_seg_sampling(self, config=None):
        db_sampler = semantic_seg_sampler.SemanticSegDataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            logger=self.logger
        )
        return db_sampler
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']

        if 'top_lidar_origin' in data_dict['scene_wise']:
            origin = data_dict['scene_wise']['top_lidar_origin']
        else:
            origin = None

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            if origin is not None:
                gt_boxes, points, origin = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points, origin=origin
                )
                data_dict['scene_wise']['top_lidar_origin'] = origin
            else:
                gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                    gt_boxes, points, origin=None
                )
        
        data_dict['object_wise']['gt_box_attr'] = gt_boxes 
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']

        if 'top_lidar_origin' in data_dict['scene_wise']:
            origin = data_dict['scene_wise']['top_lidar_origin']
            gt_boxes, points, origin = augmentor_utils.global_rotation(
                gt_boxes, points, rot_range=rot_range, origin=origin
            )
            data_dict['scene_wise']['top_lidar_origin'] = origin
        else:
            gt_boxes, points = augmentor_utils.global_rotation(
                gt_boxes, points, rot_range=rot_range, origin=None
            )

        data_dict['object_wise']['gt_box_attr'] = gt_boxes 
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']

        if 'top_lidar_origin' in data_dict['scene_wise']:
            origin = data_dict['scene_wise']['top_lidar_origin']
            gt_boxes, points, origin = augmentor_utils.global_scaling(
                gt_boxes, points, config['WORLD_SCALE_RANGE'], origin=origin
            )
            data_dict['scene_wise']['top_lidar_origin'] = origin
        else:
            gt_boxes, points = augmentor_utils.global_scaling(
                gt_boxes, points, config['WORLD_SCALE_RANGE'], origin=None
            )
        
        data_dict['object_wise']['gt_box_attr'] = gt_boxes 
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        if noise_translate_std == 0:
            return data_dict
        if 'top_lidar_origin' in data_dict['scene_wise']:
            origin = data_dict['scene_wise']['top_lidar_origin']
        else:
            origin = None

        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            if origin is not None:
                gt_boxes, points, origin = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                    gt_boxes, points, noise_translate_std, origin=origin,
                )
                data_dict['scene_wise']['top_lidar_origin'] = origin
            else:
                gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                    gt_boxes, points, noise_translate_std,
                )

        data_dict['object_wise']['gt_box_attr'] = gt_boxes 
        data_dict['point_wise']['point_xyz'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )
        
        data_dict['object_wise']['gt_box_attr'] = gt_boxes 
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']
        gt_boxes, points = augmentor_utils.local_rotation(
            gt_boxes, points, rot_range=rot_range
        )
        
        data_dict['object_wise']['gt_box_attr'] = gt_boxes
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        print('scaling')
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        points = data_dict['point_wise']['point_xyz']
        gt_boxes, points = augmentor_utils.local_scaling(
            gt_boxes, points, config['LOCAL_SCALE_RANGE']
        )
        
        data_dict['object_wise']['gt_box_attr'] = gt_boxes
        data_dict['point_wise']['point_xyz'] = points
        return data_dict
    
    def drop_labels(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.drop_labels, config=config)
        point_wise_dict = data_dict['point_wise']
        seg_labels = point_wise_dict['segmentation_label']
        ins_labels = point_wise_dict['instance_label']
        drop_ratio = config.get("DROP_RATIO", 0.0)
        drop_classes = config.get("DROP_CLASSES", [])

        for cls in drop_classes:
            ins_cls = ins_labels[seg_labels==cls]
            ins_cls = np.unique(ins_cls)
            n = ins_cls.shape[0]
            drop_cls = np.random.permutation(ins_cls)[:int(n*drop_ratio)]
            for d_cls in drop_cls:
                seg_labels[ins_labels==d_cls] = 0
        point_wise_dict['segmentation_label'] = seg_labels
        data_dict['point_wise'] = point_wise_dict
        return data_dict


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        
        data_dict['object_wise']['gt_box_attr'][:, 6] = common_utils.limit_period(
            data_dict['object_wise']['gt_box_attr'][:, 6], offset=0.5, period=2 * np.pi
        )

        return data_dict
