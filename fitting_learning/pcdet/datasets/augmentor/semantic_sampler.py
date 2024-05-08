import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from .database_sampler import DataBaseSampler
from ...datasets.waymo import (
    split_by_seg_label
)
from sklearn.neighbors import NearestNeighbors as NN

def filter_by_mask(mask, *args):
    ret_args = []
    for arg in args:
        ret_args.append(arg[mask])
    return ret_args

class SemanticSampler(DataBaseSampler):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None, visualize=False):
        super(SemanticSampler, self).__init__(root_path, sampler_cfg, class_names, logger)
        db_info_path = sampler_cfg.get('DB_INFO_PATH', None)
        self.rotate_to_face_camera = sampler_cfg.get("ROTATE_TO_FACE_CAMERA", False)
        drop_points_by_distance = self.sampler_cfg.get('DROP_POINTS_BY_DISTANCE', None)
        if drop_points_by_distance is not None:
            self.dist2npoint = {}
            for class_name, path in drop_points_by_distance.items():
                with open(path, 'rb') as fin:
                    buckets = pickle.load(fin)
                new_buckets = {}
                for d, val in buckets.items():
                    d = np.clip(d, 0, 100-0.1)
                    d_int = int(d*10)
                    new_buckets[d_int] = val
                self.dist2npoint[class_name] = new_buckets 
        else:
            self.dist2npoint = None
        
        if sampler_cfg.get('AUG_AREA', None) is None:
            raise ValueError('AUG_AREA should be specified')
        if sampler_cfg.get('AUG_SEGMENT_SOURCE', None) is None:
            raise ValueError('AUG_SEGMENT_SOURCE should be specified')
        
        self.aug_area = sampler_cfg.AUG_AREA
        self.aug_segment_source = sampler_cfg.AUG_SEGMENT_SOURCE

        self.sequence_level_semantics = sampler_cfg.get("SEQUENCE_LEVEL_SEMANTICS", None)
        if self.sequence_level_semantics is not None:
            self.aug_support_path = self.sequence_level_semantics.get('path', None)
            self.sequence_level_semantics = self.sequence_level_semantics.get('enabled', False)
            if self.aug_support_path is not None:
                aug_support_path = self.root_path.resolve() / self.aug_support_path
                with open(aug_support_path, 'rb') as fin:
                    self.aug_support_dict = pickle.load(fin)
        else:
            self.sequence_level_semantics = False 
        self.oversample_rate = sampler_cfg.get("OVERSAMPLE_RATE", 1)

        self.interaction_filter = sampler_cfg.get('INTERACTION_FILTER', None)
        self.max_num_trial = 20
        self.visualize = visualize

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

    def sample_with_fixed_number(self, class_name, sample_group, oversample_rate):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        sample_num = sample_num * oversample_rate
        if pointer + sample_num > len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
            
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict
    
    def sample_road_points(self, road_points, sample_group, oversample_rate=1):
        sample_num = int(sample_group['sample_num']) * oversample_rate
        rand_indices = np.random.choice(np.arange(road_points.shape[0]),
                                        sample_num, replace=False)
        return road_points[rand_indices], rand_indices

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

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names']
        points = data_dict['points']
        seg_labels = data_dict['seg_labels']
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

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            
            if self.rotate_to_face_camera and ('delta_angle' in info):
                angle = info['delta_angle']
                R = np.array([np.cos(angle), -np.sin(angle),
                              np.sin(angle),  np.cos(angle)]).reshape(2, 2)
                obj_points[:, :2] = obj_points[:, :2] @ R.T

            obj_points[:, :3] += info['box3d_lidar'][:3]
        
            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            # remove points if there is too much
            if self.dist2npoint is not None:
                dist = np.linalg.norm(info['box3d_lidar'][:3], ord=2).clip(0, 100-0.1)
                key = int(dist * 10)
                npoint = int(self.dist2npoint[info['name']][key])+1
                if obj_points.shape[0] > npoint:
                    rand_indices = np.random.permutation(obj_points.shape[0])[:npoint]
                    obj_points = obj_points[rand_indices]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        obj_padding_seg_labels = -np.ones((obj_points.shape[0], 2)).astype(seg_labels.dtype)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points, seg_labels = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes, seg_labels)
        points = np.concatenate([obj_points, points], axis=0)
        seg_labels = np.concatenate([obj_padding_seg_labels, seg_labels], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['seg_labels'] = seg_labels
        return data_dict, obj_points

    def remove_overlapping_boxes(self, sampled_boxes, existed_boxes):
        iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
        iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
        iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
        iou1 = iou1 if iou1.shape[1] > 0 else iou2
        valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
        return valid_mask

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask].astype(str)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict.pop('gt_boxes_mask')

        points = data_dict['points']
        seg_labels = np.copy(data_dict['seg_labels'])
        top_lidar_points = points[:seg_labels.shape[0]]
        road, sidewalk, other_obj, seg_labels = split_by_seg_label(points, seg_labels)

        tree = NN(n_neighbors=4).fit(other_obj[:, :3])
        n3rd_dist, _ = tree.kneighbors(other_obj[:, :3])
        n3rd_dist = n3rd_dist[:, 3]
        other_obj = other_obj[n3rd_dist < 1]
        
        walkable = np.concatenate([road, sidewalk], axis=0)
        non_walkable = other_obj
        non_road = np.concatenate([sidewalk, other_obj], axis=0)

        segments = {}
        segments['frame'] = dict(
            road=road,
            sidewalk=sidewalk,
            other_obj=other_obj,
            walkable=walkable,
            non_walkable=non_walkable,
            non_road=non_road
        )

        if self.sequence_level_semantics:
            sequence_name = data_dict['frame_id'][:-4]
            support = self.aug_support_dict[sequence_name]
            road = support['road']
            sidewalk = support['walkable']
            # transfer to frame coordinate system
            pose = data_dict.pop('pose').astype(np.float64)
            road = (road - pose[:3, 3]) @ pose[:3, :3]
            sidewalk = (sidewalk - pose[:3, 3]) @ pose[:3, :3]

            walkable = np.concatenate([road, sidewalk], axis=0)
            # non_walkable does not need to change
            non_road = np.concatenate([sidewalk, other_obj], axis=0)
            global_segments = dict(
                road=road,
                sidewalk=sidewalk,
                other_obj=other_obj,
                walkable=walkable,
                non_walkable=non_walkable,
                non_road=non_road
            )
            segments['sequence'] = global_segments

        if (len(road.shape) == 0) or (road.shape[0] == 0):
            return data_dict
        existed_boxes = gt_boxes
        sampled_obj_points = np.zeros((0, 3), dtype=np.float32)
        
        if self.visualize:
            from pcdet.utils.visualization import Visualizer; vis = Visualizer()
            self.vis = vis
            #vis.pointcloud('points', points[:, :3])
            for key, seg in segments.items():
                vis.pointcloud(f'{key}-other-obj', seg['other_obj'][:, :3],color=(0,0.5,0.7))
                vis.pointcloud(f'{key}-road', seg['road'][:, :3], color=(75, 75, 75))
                vis.pointcloud(f'{key}-sidewalk', seg['sidewalk'][:, :3], color=(0,0.5,0))
            #vis.pointcloud('non_walkable', non_walkable[:, :3])
            #vis.pointcloud('non_road', non_road[:, :3])
            corners = box_utils.boxes_to_corners_3d(existed_boxes)
            vis.boxes(f'gt-boxes', corners, color=(1,0,0))
        
        #import ipdb; ipdb.set_trace()
        for class_name, sample_group in self.sample_groups.items():
            key = self.aug_segment_source[class_name]
            assert key in segments, \
                f"key={key} is not a valid augmentation area, candidate: (frame, sequence)"

            candidate_area, boundary_area = self.aug_area[class_name]

            candidate_locations = segments[key][candidate_area]
            if (len(candidate_locations.shape) == 0) or (candidate_locations.shape[0] == 0):
                continue
            boundary_locations = segments[key][boundary_area]

            # compute and save distance map from each candidate point 
            # to the nearest background point
            tree = NN(n_neighbors=1).fit(boundary_locations[:, :2])
            nn_dists, _ = tree.kneighbors(candidate_locations[:, :2])
            nn_dists = nn_dists[:, 0]

            for trial in range(self.max_num_trial):
                gt_boxes = data_dict['gt_boxes']
                gt_names = data_dict['gt_names'].astype(str)
                total_valid_sampled_dict = []
                if self.limit_whole_scene:
                    num_gt = np.sum(class_name == gt_names)
                    sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
                oversample_rate = self.oversample_rate
                while int(sample_group['sample_num']) * oversample_rate > self.db_infos[class_name].__len__():
                    oversample_rate -= 1
                assert oversample_rate > 0, \
                    f"database do not contain enough examples for class {class_name}"
                if int(sample_group['sample_num']) <= 0:
                    break
                else:
                    sampled_dict = self.sample_with_fixed_number(class_name, sample_group, oversample_rate)

                    sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                    if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                        sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                    sampled_locations, sampled_indices = self.sample_road_points(candidate_locations, sample_group, oversample_rate)
                    dist2boundary = nn_dists[sampled_indices]

                    if sampled_locations.shape[0] < int(sample_group['sample_num'])*oversample_rate:
                        continue
                    
                    if self.rotate_to_face_camera:
                        src_angles = np.arctan2(sampled_boxes[:, 1], sampled_boxes[:, 0])
                        target_angles = np.arctan2(sampled_locations[:, 1], sampled_locations[:, 0]) # y, x
                        delta_angles = target_angles - src_angles

                        for i in range(len(sampled_dict)):
                            sampled_dict[i]['delta_angle'] = delta_angles[i]
                        sampled_boxes[:, -1] += delta_angles

                    sampled_boxes[:, :3] = sampled_locations
                    sampled_boxes[:, 2] += sampled_boxes[:, 5] / 2
                    for i in range(len(sampled_dict)):
                        sampled_dict[i]['box3d_lidar'][:] = sampled_boxes[i][:]

                    # remove overlapping boxes
                    valid_indices = self.remove_overlapping_boxes(sampled_boxes, existed_boxes)
                    #valid_sampled_boxes = sampled_boxes[valid_mask]
                    valid_sampled_boxes, dist2boundary = filter_by_mask(valid_indices, sampled_boxes, dist2boundary)
                    valid_sampled_dict = [sampled_dict[x] for x in valid_indices]
                    
                    if len(valid_sampled_dict) > 0:
                        # remove boxes that contain background points
                        box_2d_diameter = np.linalg.norm(valid_sampled_boxes[:, 3:5], ord=2, axis=-1) / 2
                        box_valid_mask = dist2boundary > box_2d_diameter
                        if self.interaction_filter is not None:
                            # remove box that are not interacting with radius r
                            radius = self.interaction_filter[class_name]
                            if radius > 0:
                                box_valid_mask = (box_2d_diameter + radius > dist2boundary) & box_valid_mask 

                        if not box_valid_mask.any():
                            continue
                        box_valid_indices = np.where(box_valid_mask)[0]
                        #indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        #              boundary_locations, valid_sampled_boxes
                        #          ) # [num_box, num_point]

                        #box_valid_mask = (indices.max(1) == 0).nonzero()[0]
                        valid_sampled_dict = [valid_sampled_dict[x] for x in box_valid_indices]
                        valid_sampled_boxes = valid_sampled_boxes[box_valid_indices]
                        #valid_sampled_boxes = filter_by_mask(box_valid_mask, valid_sampled_boxes)
                        
                        if valid_sampled_boxes.shape[0] > int(sample_group['sample_num']):
                            valid_sampled_boxes = valid_sampled_boxes[:int(sample_group['sample_num'])]
                            valid_sampled_dict = valid_sampled_dict[:int(sample_group['sample_num'])]

                        existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                            
                        if self.visualize:
                            corners = box_utils.boxes_to_corners_3d(valid_sampled_boxes)
                            vis.boxes(f'sampled-boxes-{class_name}-trial-{trial}', corners, color=(0,0,1))

                        total_valid_sampled_dict.extend(valid_sampled_dict)
                        
                        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
                        if total_valid_sampled_dict.__len__() > 0:
                            data_dict, obj_points = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
                            sampled_obj_points = np.concatenate([sampled_obj_points, obj_points[:, :3]], axis=0)
                            non_walkable = np.concatenate([non_walkable, obj_points[:, :3]], axis=0)
                            non_road = np.concatenate([non_road, obj_points[:, :3]], axis=0)
            
                
        if self.visualize:
            corners = box_utils.boxes_to_corners_3d(valid_sampled_boxes)
            #points = data_dict['points']
            vis.pointcloud('sampled-obj-points', sampled_obj_points[:, :3], color=(1,1,0))
            vis.show()
        
        return data_dict
