import os
import pickle
import numpy as np
import time 

from psbody.mesh import Mesh 
from smpl_webuser.serialization import load_model

import torch 
from torch_cluster import fps 
try:
    import pymesh 
except:
    pass  

from ..dataset import DatasetTemplate

class ImageFeatureDataset(DatasetTemplate):
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
        self.smpl_model_female = load_model('/home/bosun/projects/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl')
        self.smpl_model_male = load_model('/home/bosun/projects/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl')

        self.pose_thres = self.dataset_cfg.get("POSE_THRES", 0.3)
        self.remove_intersection = self.dataset_cfg.get("REMOVE_INTERSECTION", False)
        self.add_gender = self.dataset_cfg.get("ADD_GENDER", False) 
        self.random_sample = self.dataset_cfg.get("RANDOM_SAMPLE", False) 
        
        self.feature_path = self.dataset_cfg.get("FEATURE_PATH", "") 
        self.use_pose_feature = self.dataset_cfg.get("USE_POSE_FEATURE", False)
         
        self.data_path = os.path.join(self.root_path, self.split+'_pose.npy')
        self.data = np.load(self.data_path)
        self.shape_data = np.load(os.path.join(self.root_path, self.split+'_shape.npy'))
        
        self.use_all  = self.dataset_cfg.get("USE_ALL", True) 
        self.amass_val = self.dataset_cfg.get("AMASS_VAL", False)
        
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
            if not os.path.exists(os.path.join(self.root_path, 'eval_poses_fix_3.npy')):
                val_pose = np.zeros((1024, 72))
                for j in range(1024): 
                    
                    val_pose[j] = np.random.normal(self.pose_mean, self.pose_std * self.std_factor)
                np.save(os.path.join(self.root_path, 'eval_poses_fix_3.npy'), val_pose)
                print('saved eval poes', os.path.join(self.root_path, 'eval_poses_fix_3.npy'))

        if self.set_val and (self.split=='val' or not self.sample_gaussian): 
            self.data = np.load(self.val_path)
        
        # self.poses = np.load(os.path.join(self.data_path, 'poses.npy'))
        temp_mesh = Mesh(filename=self.template_path)
        scale_mesh = Mesh(filename='/home/bosun/projects/PCPerception/tools/template/smpl_template.obj')
        
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
        
        self.test_single = self.dataset_cfg.get("TEST_SINGLE", False)
        self.single_mesh_path = self.dataset_cfg.get("SINGLE_MESH_PATH", "")
        self.sing_fea_path = self.dataset_cfg.get("SINGLE_FEA_PATH", "")
    
        # load image features 
        self.load_features()
        print("loaded features",  self.shape_params.shape, self.clip_features.shape, self.dino_features.shape)
        
        
    def __len__(self):
        if self.split=='train':
            return self.num_train
        else:
            return self.num_val

    def load_features(self, ):
        if self.split=='train':
            i_range = range(11)
            j_num = self.num_train//(100*len(i_range))+1 
        else:
            i_range = range(11) 
            j_num = self.num_val//(100*len(i_range))+1 
        
        if self.random_sample:
            j_num = 100
        if self.split=='train':
            j_range = range(j_num)
            total_num = len(i_range)*j_num*100 
        else:
            j_range = range(j_num+100)
            total_num = 10000 
        
        out_path = os.path.join(self.feature_path, 'total_features_%s.npz'%(self.split)) 
        # out_path = os.path.join(self.feature_path, 'total_features_val.npz')
        if  os.path.exists(out_path):
            a = np.load(out_path)
            self.shape_params = a['shape_params']
            self.clip_features = a['clip_features']
            self.dino_features = a['dino_features'] 
            if self.use_pose_feature:
                self.pose_params = a['pose_params'] 
            return  
        print('##total', total_num)
        self.shape_params = np.zeros((total_num, 17))
        self.clip_features = np.zeros((total_num, 8, 512))
        self.dino_features = np.zeros((total_num, 8, 384))
        self.pose_params = np.zeros(1)
        if self.use_pose_feature:
            self.pose_params = np.zeros((total_num, 72)) 
            
        cnt = 0 
        # j_range = range(j_num)
        for i, _ in enumerate(i_range):
            print('loading..', i, j_num)
            if cnt*100>=total_num:
                break 
            for j in j_range:
                
                print('cnt', cnt, i, j)
                if not os.path.exists(os.path.join(self.feature_path, "%d_%d.npz"%(i, j))):
                    continue 
                try:
                    a = np.load(os.path.join(self.feature_path, "%d_%d.npz"%(i, j)))
                except:
                    print('??????', i, j)
                    continue 
                
                shape_param = a['shape_param']
                clip_fea = a['clip_features']
                dino_fea = a['dino_features']
                self.shape_params[cnt*100:(cnt+1)*100] = shape_param 
                self.clip_features[cnt*100:(cnt+1)*100] = clip_fea
                self.dino_features[cnt*100:(cnt+1)*100] = dino_fea
                
                if self.use_pose_feature:
                    self.pose_params[cnt*100:(cnt+1)*100] = a['pose_param']
                      
                cnt += 1 
                if cnt*100>=total_num:
                    break 
                
        self.shape_params = self.shape_params[:cnt*100]
        self.clip_features = self.clip_features[:cnt*100]
        self.dino_features = self.dino_features[:cnt*100]
        if self.use_pose_feature:
            self.pose_params = self.pose_params[:cnt*100]
        
        # if self.random_sample:
        #     np.savez(out_path, shape_params=self.shape_params, clip_features=self.clip_features, \
        #         dino_features=self.dino_features, pose_params=self.pose_params)
            
    def __getitem__(self, index):
        if self.random_sample:
            index = np.random.randint(0, self.shape_params.shape[0])
        
        pose_index = index 
        shape_index = index 

        tmp_shape = self.shape_params[shape_index]
        if self.test_single:
            tmp_shape = np.zeros(tmp_shape.shape)

        if tmp_shape[0]==0:
            smpl_model = self.smpl_model_male
        else:
            smpl_model = self.smpl_model_female 
            
        if not self.sample_gaussian:
            tmp_pose = self.data[pose_index]
            smpl_model.pose[:] = tmp_pose
            smpl_model.betas[:16] = tmp_shape[1:]
            vert2 = smpl_model.r 
        else:
            tmp_pose = np.random.normal(self.pose_mean, self.pose_std * self.std_factor)
            smpl_model.pose[:] = tmp_pose
            smpl_model.betas[:16] = tmp_shape[1:]
            vert2 = smpl_model.r 
            if self.remove_intersection:
                step=0
                inter_faces_ids = np.zeros((200, 2))
                while(inter_faces_ids.shape[0]>150 and step<10):
                    tmp_pose = np.random.normal(self.pose_mean, self.pose_std * self.std_factor)    
                    smpl_model.pose[:] = tmp_pose
                    smpl_model.betas[:16] = tmp_shape[1:]
                    vert2 = smpl_model.r 
                    mesh = pymesh.form_mesh(vert2, self.temp_face)
                    inter_faces_ids = pymesh.detect_self_intersection(mesh)
                    step += 1 
        
        clip_fea = self.clip_features[shape_index].reshape(-1) 
        dino_fea = self.dino_features[shape_index].reshape(-1) 
            
        temp_pose = np.zeros(smpl_model.pose.size)
        temp_pose[8] = -0.5
        temp_pose[5] = 0.5
        
        if self.use_pose_feature:
            temp_pose = self.pose_params[pose_index]
            
        smpl_model.pose[:] = temp_pose
        smpl_model.betas[:16] = tmp_shape[1:]
        vert1 = smpl_model.r 
        
        if self.test_single:
            tmp_shape = np.zeros(tmp_shape.shape[0])
            single_mesh = Mesh(filename=self.single_mesh_path)
            single_fea = np.load(self.sing_fea_path)
            clip_fea = single_fea['clip_features'].reshape(-1)
            dino_fea = single_fea['dino_features'].reshape(-1)
            vert1 = single_mesh.v 
            source_face = single_mesh.f 
        else:
            source_face = self.temp_face 
            
        vert2 = vert2 - self.min1
        vert2 = vert2 / (self.max1_1[1] - self.min1_1[1])
        vert2 = (vert2 - (self.max1_2 - self.min1_2)/2)*self.scale_factor 
        
        vert1 = vert1 - self.min1
        vert1 = vert1 / (self.max1_1[1] - self.min1_1[1])
        vert1 = (vert1 - (self.max1_2 - self.min1_2)/2)*self.scale_factor 
        
        point_wise_dict = dict(
            pose_param = tmp_pose, 
            shape_param = tmp_shape,
            shape_points = vert2,
            temp_points = vert1, 
            clip_feature = clip_fea,
            dino_feature = dino_fea, 
            source_face = source_face.astype(np.int32), 
        )
        scene_wise_dict = dict(
            idx = index,
        )

        data_dict = dict(
            point_wise = point_wise_dict,
            scene_wise = scene_wise_dict,
        )

        return data_dict

    