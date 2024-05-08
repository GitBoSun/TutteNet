import pickle

import os
import copy
import numpy as np
import torch
import SharedArray as SA
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, sa_utils
#from ...models.visualizers import PolyScopeVisualizer
from ...config import cfg_from_yaml_file, cfg
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import (
    points_in_boxes_cpu
)

NAME2LABEL = {
    'Vehicle': 1,
    'Cyclist': 2,
    'Pedestrian': 3,
}

class SemanticSegDataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, logger=None):
        self.root_path = root_path
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        self.aug_classes = sampler_cfg['AUG_CLASSES']
        self.disable_after_epoch = sampler_cfg.get('DISABLE_AFTER_EPOCH', 100)
        self.data_tag = 'waymo_seg_with_r2_top'
        db_info_paths = sampler_cfg["DB_INFO_PATH"]
        
        self.box_translation = {i: 0 for i in range(50)}
        for i in range(1, 5):
            self.box_translation[i] = 1
        for i in range(5, 7):
            self.box_translation[i] = 3
        for i in range(7, 8):
            self.box_translation[i] = 2

        self.db_infos = None
        for db_info_path in db_info_paths:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as fin:
                db_infos = pickle.load(fin)
                if self.db_infos is None:
                  self.db_infos = db_infos
                else:
                  for key in db_infos.keys():
                    self.db_infos[key] = np.concatenate([self.db_infos[key], db_infos[key]],
                                                        axis=0)

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
        
        self.num_infos_by_cls = np.zeros(23).astype(np.int32)
        self.indices_by_cls = {}
        for i in range(23):
            self.num_infos_by_cls[i] = (self.db_infos['cls_of_info'] == i).sum()
            self.indices_by_cls[i] = np.where(self.db_infos['cls_of_info'] == i)[0]

        for data_type in ['db_point_feat_label']:
            for split in ['training']:
                sa_utils._allocate_data(self.data_tag, split, data_type, self.root_path)
            if logger is not None:
                logger.info(f"Allocated database into Shared Memory")
        self.sample_groups = {}
        self.sample_class_num = {}

        for sample_group in sampler_cfg["SAMPLE_GROUPS"]:
            cls = sample_group['cls']
            sample_num = sample_group['num']
            self.sample_groups[cls] = {
                'sample_num': sample_num,
                'pointer': self.num_infos_by_cls[cls],
                'indices': np.arange(self.num_infos_by_cls[cls]),
                'num_trial': sample_group['num_trial'],
                'scene_limit': sample_group['scene_limit']
            }
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.logger is not None:
            self.logger.info(f"Setting epoch to {epoch}, enabled={self.epoch <= self.disable_after_epoch}")

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        pass
    
    def filter_by_min_points(self, db_infos, min_num_points):
        mask = db_infos['num_points'] >= min_num_points
        new_db_infos = {}
        for key in db_infos.keys():
            new_db_infos[key] = db_infos[key][mask]
        if self.logger is not None:
            self.logger.info('Database filter by min points %d => %d' %
                (mask.shape[0], mask.sum()))
        return new_db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = sample_group['num_trial'], sample_group['pointer'], sample_group['indices']
        if pointer >= self.num_infos_by_cls[class_name]:
            indices = np.random.permutation(self.num_infos_by_cls[class_name])
            pointer = 0

        sampled_indices = [self.indices_by_cls[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        sampled_dict = []
        for index in sampled_indices:
            s_dict = dict(
                num_points=self.db_infos['num_points'][index],
                trans_z=self.db_infos['trans_z'][index],
                #path=self.db_infos['paths'][index],
                support_cls=self.db_infos['support_cls'][index],
                box3d=self.db_infos['box3d'][index],
                index=index,
            )
            if np.abs(s_dict['box3d']).sum() < 1e-2:
                s_dict['box3d'] = None
            sampled_dict.append(s_dict)
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    def sample_candidate_locations(self, cls_points_dict, support_classes):
        """
        Args:
            points [N, 3+C]
            seg_cls_labels [N]
            support_class [M]
        Returns:
            valid [M1]
            locations [M1, 3]
        """
        valid = []
        locations = []
        for i, support_class in enumerate(support_classes):
            cls_points = cls_points_dict[support_class]
            if cls_points.shape[0] == 0:
                continue
            index = np.random.randint(0, cls_points.shape[0])
            locations.append(cls_points[index, :3])
            valid.append(i)
        return valid, np.array(locations)

    def __call__(self, data_dict):
        if self.epoch > self.disable_after_epoch:
            return data_dict

        coord = data_dict['point_wise']['point_xyz']
        feat = data_dict['point_wise']['point_feat']
        label = data_dict['point_wise']['segmentation_label']
        if 'instance_label' in data_dict['point_wise']:
            seg_inst_labels = data_dict['point_wise']['instance_label']
        else:
            seg_inst_labels = np.zeros_like(seg_cls_labels).astype(np.int32)

        points = np.concatenate([coord, feat], axis=1)
        original_points = np.copy(points)
        seg_cls_labels = label
        assert seg_cls_labels.shape[0] == seg_inst_labels.shape[0]

        foreground_mask = np.zeros_like(seg_cls_labels).astype(bool)
        for i in range(1, 17):
            foreground_mask = foreground_mask | (seg_cls_labels == i)
        foreground_points = points[np.where(foreground_mask)[0]]
        if data_dict['object_wise'].get('gt_box_attr', None) is not None:
            num_boxes = data_dict['object_wise']['gt_box_attr'].shape[0]
            existed_boxes = np.zeros((num_boxes, 10))
            if num_boxes > 0:
                existed_boxes[:, :7] = data_dict['object_wise']['gt_box_attr']
                data_dict['object_wise']['gt_box_cls_label'] = np.array([
                        NAME2LABEL[ll] if ll in NAME2LABEL else ll
                        for ll in data_dict['object_wise']['gt_box_cls_label']
                    ]).astype(np.int32)
                existed_boxes[:, 7] = data_dict['object_wise']['gt_box_cls_label']
                #assert (data_dict['object_wise']['gt_box_cls_label'] < 4).all()
                if 'difficulty' in data_dict['object_wise']:
                    existed_boxes[:, 8] = data_dict['object_wise']['difficulty']
                existed_boxes[:, 9] = data_dict['object_wise']['num_points_in_gt']
        else:
            existed_boxes = np.zeros((0, 10))
        cls_points_dict = {i: points[seg_cls_labels == i, :3] for i in self.aug_classes + [18, 21, 22]}

        keys = np.array([k for k in self.sample_groups.keys()]).astype(np.int32)
        random_order = np.random.permutation(keys.shape[0])
        keys = keys[random_order]
        for fg_cls in keys:
            sample_group = self.sample_groups[fg_cls]
            aug_point_list = []
            aug_seg_cls_label_list = []
            aug_seg_inst_label_list = []
            aug_box_list = []

            if sample_group['scene_limit'] > 0:
                try:
                    num_instance = np.unique(seg_inst_labels[seg_cls_labels == fg_cls]).shape[0]
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    print(e)
                sample_group['sample_num'] = sample_group['scene_limit'] - num_instance

            if sample_group['sample_num'] > 0:
                sampled_dict = self.sample_with_fixed_number(fg_cls, sample_group)

                # sample locations
                support_classes = [d['support_cls'] for d in sampled_dict]
                valid, candidate_locations = self.sample_candidate_locations(cls_points_dict, support_classes)
                if len(valid) == 0:
                    continue
                sampled_dict = [sampled_dict[i] for i in valid]
                for sampled_d, loc in zip(sampled_dict, candidate_locations):
                    #path = sampled_d['path']
                    trans_z = -sampled_d['trans_z']
                    aug_points = SA.attach("shm://waymo_{}.db_point_feat_label".format(sampled_d['index'])).copy()
                    #aug_points = np.load(path)
                    aug_seg_cls_labels = aug_points[:, -1].astype(np.int32)
                    assert aug_points.shape[-1] == 7 # (x, y, z, range, intensity, elongation, label)
                    aug_points = aug_points[:, [0,1,2,4,5,3]] # drop label
                    aug_points[:, 3] = np.tanh(aug_points[:, 3]) # normalize intensity
                    aug_points[:, 5] = aug_points[:, 5] / 75.0 # normalize intensity
                    low = aug_points.mean(0)
                    trans = loc - low[:3]
                    trans[2] -= trans_z
                    aug_points[:, :3] += trans
                    # estimate or reuse bounding boxes
                    if sampled_d.get('box3d', None) is not None:
                        #print(fg_cls, round(sampled_d['box3d'][7]))
                        box = np.zeros(10)
                        box[:sampled_d['box3d'].shape[0]] = sampled_d['box3d']
                        box[7] = self.box_translation[round(sampled_d['box3d'][7])]
                        box[:3] += trans
                        box[9] = aug_points.shape[0]
                        aug_box_list.append(box)
                    else:
                        box = np.zeros(10)
                        box[:3] = (aug_points.max(0)[:3] + aug_points.min(0)[:3]) / 2
                        box[3:6] = (aug_points.max(0) - aug_points.min(0))[:3] + 0.05
                        box[7] = self.box_translation[fg_cls]
                        box[8] = 0
                        box[9] = aug_points.shape[0]
                        aug_box_list.append(box)
                    #print(fg_cls, np.unique(aug_seg_cls_labels), box[7], None if sampled_d.get('box3d', None) is None else round(sampled_d['box3d'][7]))
                    # low + trans = loc - trans_z
                    aug_point_list.append(aug_points)
                    aug_seg_cls_label_list.append(aug_seg_cls_labels)
                # estimate bounding boxes
                aug_boxes = torch.from_numpy(np.stack(aug_box_list, axis=0)).view(-1, 10).float().numpy()

                # reject by collision
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(aug_boxes[:, 0:7], existed_boxes[:, 0:7]).astype(np.float32)
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(aug_boxes[:, 0:7], aug_boxes[:, 0:7])
                iou2[range(aug_boxes.shape[0]), range(aug_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                box_valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                point_mask = points_in_boxes_cpu(foreground_points[::5, :3], aug_boxes[:, :7]).any(-1)  # [num_boxes]
                box_valid_mask = (box_valid_mask & (point_mask == False)).nonzero()[0]

                aug_boxes = aug_boxes[box_valid_mask]
                aug_point_list = [aug_point_list[i] for i in box_valid_mask]
                aug_seg_cls_label_list = [aug_seg_cls_label_list[i] for i in box_valid_mask]

                if len(aug_point_list) > 0:
                    if len(aug_point_list) > sample_group['sample_num']:
                        diff = len(aug_point_list) - sample_group['sample_num']
                        aug_point_list = aug_point_list[:sample_group['sample_num']]
                        aug_seg_cls_label_list = aug_seg_cls_label_list[:sample_group['sample_num']]
                        aug_boxes = aug_boxes[:sample_group['sample_num']]
                    aug_points = torch.from_numpy(np.concatenate(aug_point_list, axis=0))
                    aug_seg_cls_labels = torch.from_numpy(np.concatenate(aug_seg_cls_label_list, axis=0))
                    # update
                    foreground_points = np.concatenate([foreground_points, aug_points], axis=0)
                    points = np.concatenate([points, aug_points], axis=0)
                    assert seg_cls_labels.shape[0] == seg_inst_labels.shape[0]
                    seg_cls_labels = np.concatenate([seg_cls_labels, aug_seg_cls_labels], axis=0)
                    seg_inst_labels = np.concatenate([seg_inst_labels, torch.zeros_like(aug_seg_cls_labels) - 1],
                                                     axis=0)
                    assert seg_cls_labels.shape[0] == seg_inst_labels.shape[0]
                    existed_boxes = np.concatenate([existed_boxes, aug_boxes], axis=0)

        data_dict['point_wise']['point_xyz'] = points[:, :3]
        data_dict['point_wise']['point_feat'] = points[:, 3:]
        assert np.abs(original_points - points[:original_points.shape[0], :]).max() < 1e-3
        if 'point_sweep' in data_dict['point_wise']:
            data_dict['point_wise']['point_sweep'] = np.concatenate([data_dict['point_wise']['point_sweep'],
                                                                     np.zeros((points.shape[0] - original_points.shape[0], 1))],
                                                                     axis=0)
        data_dict['point_wise']['segmentation_label'] = seg_cls_labels
        if 'instance_label' in data_dict['point_wise']:
            data_dict['point_wise']['instance_label'] = seg_inst_labels
        data_dict['object_wise']['gt_box_attr'] = existed_boxes[:, :7] 
        data_dict['object_wise']['gt_box_cls_label'] = existed_boxes[:, 7].astype(np.int32)
        data_dict['object_wise']['difficulty'] = existed_boxes[:, 8].astype(np.int32)
        data_dict['object_wise']['num_points_in_gt'] = existed_boxes[:, 9].astype(np.int32)
        mask = data_dict['object_wise']['gt_box_cls_label'] > 0
        for key in ['augmented', 'obj_ids']:
            if key in data_dict['object_wise']:
                data_dict['object_wise'].pop(key)
        for key in ['is_foreground']:
            if key in data_dict['point_wise']:
                data_dict['point_wise'].pop(key)
        data_dict['object_wise'] = common_utils.filter_dict(data_dict['object_wise'], mask)
        
        return data_dict
