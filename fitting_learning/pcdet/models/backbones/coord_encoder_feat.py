import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self,  planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.fc1 = nn.Linear(planes, planes, bias=True) 
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CoordinateEncoderFeature(nn.Module):
    def __init__(self, mesh, bound_verts, model_cfg, runtime_cfg):
        super().__init__()
        scale = runtime_cfg.get("scale", 1.0)

        self.model_cfg = model_cfg
        # input_channels = model_cfg.get("INPUT_CHANNEL", 5) 
        self.dino_channel = 384 * 8 
        self.clip_channel = 512 * 8 
        self.dino_encode_channel = model_cfg.get("DINO_ENCODE_CHANNEL", 128)
        self.clip_encode_channel = model_cfg.get("CLIP_ENCODE_CHANNEL", 128)  
        input_channels = self.dino_encode_channel + self.clip_encode_channel + 72 
        
        self.mesh = mesh
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.edge_centers = (self.vertices[self.edges[0]] + self.vertices[self.edges[1]])/2
        self.bound_verts = self.vertices[bound_verts]
        self.use_positional_encoding = model_cfg.get("USE_POSITIONAL_ENCODING", False)
        self.encoding_length = model_cfg.get("ENCODING_LENGTH", 10) 
        
        self.use_pose_positional_encoding = model_cfg.get("USE_POSE_POSITIONAL_ENCODING", False)
        self.pose_encoding_length = model_cfg.get("POSE_ENCODING_LENGTH", 10) 
        
        self.predict_normal = runtime_cfg["predict_normal"] 
        self.use_shape = runtime_cfg["use_shape"] 
        self.add_gender = runtime_cfg["add_gender"]

        input_channels_or = input_channels
        
        if  self.use_pose_positional_encoding:
            input_channels = input_channels * self.pose_encoding_length*2 
        
        if  self.use_positional_encoding:
            self.edge_centers = self.positional_encoding(self.edge_centers, L=self.encoding_length)
            self.bound_verts = self.positional_encoding(self.bound_verts, L=self.encoding_length)
            input_channels += self.encoding_length * 4
        else:
            input_channels += 2 
    
        self.num_edges = runtime_cfg["num_edges"]
        self.num_bound_verts = runtime_cfg["num_bound_verts"]
        self.rotate_angle = runtime_cfg["rotate_angle"]
        self.fix_bound = runtime_cfg["fix_bound"]
        
        self.append_bn = model_cfg.get("APPEND_BN", False)  
        self.skip_connection = model_cfg.get("SKIP_CONNECTION", False) 
        self.repeat_mlp = model_cfg.get("REPEAT_MLP", False) 
        
        self.fc1_dim = model_cfg.get("FC1_DIM", 64)
        self.fc1_dim_fea = model_cfg.get("FC1_DIM_FEA", 64)
            
        self.num_layers = runtime_cfg["num_layers"]
        # self.num_innder_layer = model_cfg.get("NUM_INNER_LAYER", 3) 
        self.num_inner_layer = runtime_cfg["num_inner_layers"]


        fc_cfg = [self.fc1_dim, ] 
        res_cfg = [self.fc1_dim, self.fc1_dim, self.fc1_dim, ]
        # res_cfg = [self.fc1_dim, self.fc1_dim]
        self.fc_w = self.make_residual_layers(fc_cfg, res_cfg, input_channels, self.num_layers*self.num_inner_layer)
        self.fc_a = self.make_residual_layers(fc_cfg, res_cfg, input_channels, self.num_layers*self.num_inner_layer,) 
        
        fc_cfg = [self.fc1_dim_fea*2, self.fc1_dim_fea, self.fc1_dim_fea//2]
        self.clip_encoder = self.make_residual_layers(fc_cfg, [], self.clip_channel, self.clip_encode_channel,) 
        self.dino_encoder = self.make_residual_layers(fc_cfg, [], self.dino_channel, self.dino_encode_channel,)  
        
        if self.predict_normal:
            fc_cfg = [self.fc1_dim//2, ] 
            res_cfg = [self.fc1_dim//2, self.fc1_dim//2, ]
            self.fc_n = self.make_residual_layers(fc_cfg, res_cfg, input_channels_or, self.num_layers*self.num_inner_layer*3,) 
        
        # self.fc_w = self.make_fc_layers([ self.fc1_dim,self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim*4, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim,], input_channels, self.num_layers*self.num_innder_layer,)
        # self.fc_a = self.make_fc_layers([ self.fc1_dim,self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim*4, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim ], input_channels, self.num_layers*self.num_innder_layer,) 
    
     
    def make_residual_layers(self, fc_cfg, res_cfg, input_channels, output_channels, append_bn=False):
        fc_layers = []
        c_in = input_channels
        
        for k in range(0, len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=True),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        
        for k in range(0, len(res_cfg)):
            fc_layers.extend([BasicBlock(c_in,)])
            c_in = res_cfg[k]
        
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def positional_encoding(self,input_xy, L=10):
        # [N, 2]
        x = input_xy[:, 0]
        y = input_xy[:, 1]

        encoding = torch.zeros((input_xy.shape[0], L, 2, 2)).cuda()
        for i in range(L):
            encoding[:, i, 0, 0] = torch.sin(x*torch.pi*(2**i))
            encoding[:, i, 1, 0] = torch.cos(x*torch.pi*(2**i))
            encoding[:, i, 0, 1] = torch.sin(y*torch.pi*(2**i))
            encoding[:, i, 1, 1] = torch.cos(y*torch.pi*(2**i))

        encoding = encoding.view(input_xy.shape[0], L*2*2)
        return encoding 

    def positional_encoding_pose(self, pose, L=10):
        # [N, 2]
        encoding = torch.zeros((pose.shape[0], pose.shape[1], L, 2)).cuda()
        for i in range(L):
            encoding[:,:, i, 0] = torch.sin(pose*torch.pi*(2**i))
            encoding[:,:,i, 1] = torch.cos(pose*torch.pi*(2**i))
            
        encoding = encoding.view(pose.shape[0], pose.shape[1]*L*2)
        return encoding 
    
    def forward(self, batch_dict):

        clip_features = batch_dict['clip_feature'].float()
        dino_features = batch_dict['dino_feature'].float()
        clip_out = self.clip_encoder(clip_features)
        dino_out = self.dino_encoder(dino_features)
        input_param = torch.cat((clip_out, dino_out, batch_dict['pose_param'].float()), dim=1).float() 
        
        batch_size = input_param.shape[0]
        # import pdb; pdb.set_trace()
        
        # [B,1, 72] [1, M, 2]
        w_input = torch.cat((input_param.unsqueeze(1).repeat(1, self.num_edges,1),\
            self.edge_centers.unsqueeze(0).repeat(batch_size, 1,1)), dim=2) # [B,M,74]
        w_input = w_input.view(batch_size*self.num_edges, -1)

        a_input = torch.cat((input_param.unsqueeze(1).repeat(1, self.num_bound_verts,1),\
            self.bound_verts.unsqueeze(0).repeat(batch_size, 1,1)), dim=2) # [B,M,74]
        a_input = a_input.view(batch_size*self.num_bound_verts, -1)
        
        w_pred = self.fc_w(w_input) # [B*M, num_layer*num_inner_layer]
        w_pred = w_pred.view(batch_size, self.num_edges, self.num_layers, self.num_inner_layer)
        
        angle_pred = self.fc_a(a_input)
        angle_pred = angle_pred.view(batch_size, self.num_bound_verts, self.num_layers, self.num_inner_layer) 
        
        for i in range(self.num_layers):
            var_pred = torch.cat((w_pred[:,:,i,:], angle_pred[:,:,i,:]), dim=1)          
            batch_dict['var_pred%d'%(i)] = var_pred
        
        if self.predict_normal:
            normal_pred = self.fc_n(input_param)
            normal_pred = normal_pred.view(batch_size, -1, 3)
            normal_pred = F.normalize(normal_pred, dim=2)
            batch_dict['normal_pred'] = normal_pred
        return batch_dict
