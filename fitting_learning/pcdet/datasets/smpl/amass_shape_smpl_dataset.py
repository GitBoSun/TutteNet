import os
import pickle
import numpy as np


from psbody.mesh import Mesh 
from smpl_webuser.serialization import load_model

import torch 
from torch_cluster import fps 
try:
    import pymesh 
except:
    pass 

from ..dataset import DatasetTemplate

class SMPLAMASSShapeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, training=training, root_path=root_path, logger=logger,
        )
        self.num_train = self.dataset_cfg.get("NUM_TRAIN", 5000)
        self.num_val = self.dataset_cfg.get("NUM_VAL", 1000)
    
        self.dataset_name = self.dataset_cfg.DATASET
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.data_path = os.path.join(self.root_path, self.split)
        self.template_path = self.dataset_cfg.get("TEMPLATE_PATH", "")
        self.fps_ratio = self.dataset_cfg.get("FPS_RATIO", 0.01)
        self.scale_factor  = self.dataset_cfg.get("SCALE_FACTOR", 1.4) 
        self.smpl_model_female = load_model('/home/bos/projects/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl')
        self.smpl_model_male = load_model('/home/bos/projects/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl')

        self.pose_thres = self.dataset_cfg.get("POSE_THRES", 0.3)
        self.remove_intersection = self.dataset_cfg.get("REMOVE_INTERSECTION", False)
        self.add_gender = self.dataset_cfg.get("ADD_GENDER", False) 
        self.generate_simple = self.dataset_cfg.get("GENERATE_SIMPLE", False)
        
        self.data_path = os.path.join(self.root_path, self.split+'_pose.npy')
        self.data = np.load(self.data_path)
        self.shape_data = np.load(os.path.join(self.root_path, self.split+'_shape.npy'))
        
        self.use_all  = self.dataset_cfg.get("USE_ALL", True) 
        self.amass_val = self.dataset_cfg.get("AMASS_VAL", True)
        
        self.set_val = self.dataset_cfg.get("SET_VAL", False)
        self.val_path =  self.dataset_cfg.get("VAL_PATH", "") 

        self.sample_gaussian = self.dataset_cfg.get("SAMPLE_GAUSSIAN", False)
        self.std_factor  = self.dataset_cfg.get("STD_FACTOR", 1.0)
        self.shape_std_factor  = self.dataset_cfg.get("SHAPE_STD_FACTOR", 1.0)
        
        if self.sample_gaussian:
            amass_data =  np.load(os.path.join(self.root_path, 'train_pose.npy'))
            self.pose_mean = amass_data.mean(0)
            self.pose_std = np.std(amass_data, axis=0)
            
            amass_shape = np.load(os.path.join(self.root_path, 'train_shape.npy'))
            self.shape_mean = amass_shape.mean(0)
            self.shape_std = np.std(amass_shape, axis=0)
            print("## amass_std", self.std_factor)
        
        if self.set_val and (self.split=='val' or not self.sample_gaussian): 
            self.data = np.load(self.val_path)
        
        self.index_mapping = [4, 28, 72, 84, 140,  168, 236, 240, ]
        # self.poses = np.load(os.path.join(self.data_path, 'poses.npy'))
        temp_mesh = Mesh(filename=self.template_path)
        scale_mesh = Mesh(filename='/home/bos/projects/PCPerception/tools/template/smpl_template.obj')
        
        vert1 = scale_mesh.v 
        self.min1 = vert1.min(0)
        vert1 = vert1 - self.min1
        self.max1_1 = vert1.max(0)
        self.min1_1 = vert1.min(0)
        vert1 = vert1 / (self.max1_1[1] - self.min1_1[1])
        self.min1_2 = vert1.min(0) 
        self.max1_2 = vert1.max(0) 
        vert1 = (vert1 - (self.max1_2 -self.min1_2)/2)*self.scale_factor 
        
        vert2 = temp_mesh.v 
        vert2 = vert2 - self.min1
        vert2 = vert2 / (self.max1_1[1] - self.min1_1[1])
        vert2 = (vert2 - (self.max1_2 - self.min1_2)/2)*self.scale_factor 
        
        self.temp_vert = torch.from_numpy(vert2).cuda().unsqueeze(0) 
        batch = torch.zeros(self.temp_vert.shape[1]).long().cuda()
        self.fps_ids = fps(self.temp_vert[0], batch, ratio=self.fps_ratio, random_start=False).cpu().numpy()
        self.temp_face = temp_mesh.f  
        
    def __len__(self):
        if self.use_all:
            return len(self.data)
        
        if self.split=='train':
            return self.num_train
        else:
            return self.num_val

    def __getitem__(self, index):
        
        # if self.split=='val' and self.amass_val:
        #     index = index * 20
       
        # if self.split=='val' and not self.sample_gaussian:
        #     index = self.index_mapping[index]
        tmp_num = np.random.random() 
        if tmp_num>=0.5 and self.add_gender:
            smpl_model = self.smpl_model_male
            shape_add = np.array([0]) 
        else:
            smpl_model = self.smpl_model_female 
            shape_add = np.array([1]) 
        
        if self.generate_simple: 
            tmp_pose = np.zeros(smpl_model.pose.size)
            tmp_pose[50] = -np.random.uniform(0,0.45*np.pi)
            tmp_pose[53] = -tmp_pose[50] 
            tmp_pose[8] = -np.random.uniform(0,0.3*np.pi)
            tmp_pose[5] = -tmp_pose[8] 
            tmp_shape = np.random.normal(self.shape_mean, self.shape_std * self.shape_std_factor) 
            
            smpl_model.pose[:] = tmp_pose
            smpl_model.betas[:16] = tmp_shape
            vert2 = smpl_model.r 
            
            if self.remove_intersection:
                step=0
                inter_faces_ids = np.zeros((200, 2))
                while(inter_faces_ids.shape[0]>150 and step<10):
                    tmp_pose = np.zeros(smpl_model.pose.size)
                    tmp_pose[50] = -np.random.uniform(0,0.45*np.pi)
                    tmp_pose[53] = -tmp_pose[50] 
                    tmp_pose[8] = -np.random.uniform(0,0.3*np.pi)
                    tmp_pose[5] = -tmp_pose[8] 
                    tmp_shape = np.random.normal(self.shape_mean, self.shape_std * self.shape_std_factor) 
            
                    smpl_model.pose[:] = tmp_pose
                    smpl_model.betas[:16] = tmp_shape
                    vert2 = smpl_model.r 
                    mesh = pymesh.form_mesh(vert2, self.temp_face)
                    inter_faces_ids = pymesh.detect_self_intersection(mesh)
                    step += 1 
             
        elif (not self.sample_gaussian) or (self.split=='val' and self.amass_val) or \
                (self.split=='val' and self.set_val):
            tmp_pose = self.data[index]
            tmp_shape = self.shape_data[index]
            smpl_model.pose[:] = tmp_pose
            smpl_model.betas[:16] = tmp_shape
            vert2 = smpl_model.r 
        else: 
            tmp_pose = np.random.normal(self.pose_mean, self.pose_std * self.std_factor)
            tmp_shape = np.random.normal(self.shape_mean, self.shape_std * self.shape_std_factor) 
            
            smpl_model.pose[:] = tmp_pose
            smpl_model.betas[:16] = tmp_shape
            vert2 = smpl_model.r 
            if self.remove_intersection:
                step=0
                inter_faces_ids = np.zeros((200, 2))
                while(inter_faces_ids.shape[0]>150 and step<10):
                    tmp_pose = np.random.normal(self.pose_mean, self.pose_std * self.std_factor)    
                    tmp_shape = np.random.normal(self.shape_mean, self.shape_std * self.shape_std_factor)                 
                    smpl_model.pose[:] = tmp_pose
                    smpl_model.betas[:16] = tmp_shape
                    vert2 = smpl_model.r 
                    mesh = pymesh.form_mesh(vert2, self.temp_face)
                    inter_faces_ids = pymesh.detect_self_intersection(mesh)
                    step += 1 
                # print(step)

        temp_pose = np.zeros(smpl_model.pose.size)
        temp_pose[8] = -0.5
        temp_pose[5] = 0.5
        smpl_model.pose[:] = temp_pose
        smpl_model.betas[:16] = tmp_shape
        vert1 = smpl_model.r 

        vert2 = vert2 - self.min1
        vert2 = vert2 / (self.max1_1[1] - self.min1_1[1])
        vert2 = (vert2 - (self.max1_2 - self.min1_2)/2)*self.scale_factor 
        
        vert1 = vert1 - self.min1
        vert1 = vert1 / (self.max1_1[1] - self.min1_1[1])
        vert1 = (vert1 - (self.max1_2 - self.min1_2)/2)*self.scale_factor 
        
        if self.add_gender:
            tmp_shape = np.concatenate((shape_add, tmp_shape))

        point_wise_dict = dict(
            pose_param = tmp_pose, 
            shape_param = tmp_shape,
            shape_points = vert2,
            temp_points = vert1, 
        )
        scene_wise_dict = dict(
            idx = index,
        )

        data_dict = dict(
            point_wise = point_wise_dict,
            scene_wise = scene_wise_dict,
        )

        return data_dict

    