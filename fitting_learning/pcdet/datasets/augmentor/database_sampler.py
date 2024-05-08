import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            
        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
        self.seg_label_map = sampler_cfg.get('SEG_LABEL_MAP', None)
        self.used_features = sampler_cfg.get("USED_FEATURE_LIST", None)
        
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
        
        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:

            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)
            
        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, sampled_seg_labels, total_valid_sampled_dict):
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        gt_names = data_dict['object_wise']['gt_box_cls_label']
        point_xyz = data_dict['point_wise']['point_xyz']
        point_feat = data_dict['point_wise']['point_feat']
        points = np.concatenate([point_xyz, point_feat], axis=-1)
        seg_cls_labels = data_dict['point_wise']['segmentation_label']
        seg_inst_labels = data_dict['point_wise']['instance_label']

        #seg_inst_labels = data_dict.get('seg_inst_labels', None)
        #seg_cls_labels = data_dict.get('seg_cls_labels', None)
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None 
        
        if 'segmentation_label' in data_dict['point_wise']:
            obj_seg_cls_labels_list = []
        if 'instance_label' in data_dict['point_wise']:
            obj_seg_inst_labels_list = []
            max_instance_label = seg_inst_labels.max()
        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']
                # [x, y, z, intensity, elongation, range, rimage_w, rimage_h]
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape([-1, 8])
                obj_points = obj_points[:, self.used_features]

            obj_points[:, :, :3] += info['box3d_lidar'][:3].reshape((1, 1, -1))

            if 'segmentation_label' in data_dict['point_wise']:
                obj_seg_cls_labels = np.full(obj_points.shape[0],
                                             info['seg_label'],                                         
                                             #self.seg_label_map[info['name']],
                                             dtype=seg_cls_labels.dtype)
                obj_seg_cls_labels_list.append(obj_seg_cls_labels)

            if 'instance_label' in data_dict['point_wise']:
                obj_seg_inst_labels = np.full(obj_points.shape[0],
                                              max_instance_label+1,
                                              dtype=seg_inst_labels.dtype)
                obj_seg_inst_labels_list.append(obj_seg_inst_labels)
                max_instance_label += 1

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )

        data_dict['point_wise'] = box_utils.remove_points_in_boxes3d(data_dict['point_wise'],
                                                                     large_sampled_gt_boxes)
        if 'instance_label' in data_dict['point_wise']:
            obj_seg_inst_labels = np.concatenate(obj_seg_inst_labels_list, axis=0)
            seg_inst_labels = data_dict['point_wise']['instance_label']
            data_dict['point_wise']['instance_label'] = np.concatenate([obj_seg_inst_labels, seg_inst_labels], axis=0)

        if 'segmentation_label' in data_dict['point_wise']:
            obj_seg_cls_labels = np.concatenate(obj_seg_cls_labels_list, axis=0)
            seg_cls_labels = data_dict['point_wise']['segmentation_label']
            data_dict['point_wise']['segmentation_label'] = np.concatenate([obj_seg_cls_labels, seg_cls_labels], axis=0)
        
        obj_points = obj_points[:, 0, :6]

        points = np.concatenate([obj_points, np.concatenate([data_dict['point_wise']['point_xyz'], data_dict['point_wise']['point_feat']], axis=-1)], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0).astype(str)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        augmented = np.concatenate([data_dict['object_wise']['augmented'],
                                    np.ones(sampled_gt_boxes.shape[0], dtype=bool)], axis=0)
        obj_ids = np.concatenate([data_dict['object_wise']['obj_ids'],
                                  np.array(['augmented' for i in range(sampled_gt_boxes.shape[0])]).astype(str)],
                                 axis=0)
        data_dict['object_wise']['gt_box_attr'] = gt_boxes
        data_dict['object_wise']['gt_box_cls_label'] = gt_names
        data_dict['object_wise']['augmented'] = augmented
        data_dict['object_wise']['obj_ids'] = obj_ids
        assert gt_boxes.shape[0] == gt_names.shape[0]
        if 'sweep' in data_dict['point_wise']:
            sweep = np.concatenate([-np.ones((obj_points.shape[0], 1), dtype=np.int32),
                                    data_dict['point_wise']['sweep']], axis=0)
            data_dict['point_wise']['sweep'] = sweep
        if 'sweep' in data_dict['object_wise']:
            sweep = np.concatenate([-np.ones((sampled_gt_boxes.shape[0], 1), dtype=np.int32),
                                    data_dict['object_wise']['sweep']], axis=0)
            data_dict['object_wise']['sweep'] = sweep
        data_dict['point_wise']['point_xyz'] = points[:, :3]
        data_dict['point_wise']['point_feat'] = points[:, 3:]

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['object_wise']['gt_box_attr']
        gt_names = data_dict['object_wise']['gt_box_cls_label'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        existed_seg_labels = np.zeros(gt_boxes.shape[0], np.int32)


        for class_name, sample_group in self.sample_groups.items():
            
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)

            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)
                sampled_seg = np.stack([x['seg_label'] for x in sampled_dict]).astype(np.int32)
                # print(sampled_dict)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                valid_sampled_seg = sampled_seg[valid_mask]


                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                existed_seg_labels = np.concatenate((existed_seg_labels, valid_sampled_seg), axis=0)         
                total_valid_sampled_dict.extend(valid_sampled_dict)
                
        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        sampled_seg_labels = existed_seg_labels[gt_boxes.shape[0]:]

        
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, sampled_seg_labels, total_valid_sampled_dict)
        for key in ['augmented', 'obj_ids', 'num_points_in_gt']:
            if key in data_dict['object_wise']:
                data_dict['object_wise'].pop(key)
        for key in ['is_foreground', 'point_sweep']:
            if key in data_dict['point_wise']:
                data_dict['point_wise'].pop(key)
        assert 'Sign' not in data_dict['object_wise']['gt_box_cls_label']

        return data_dict
