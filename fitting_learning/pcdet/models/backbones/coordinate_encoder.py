

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateEncoder(nn.Module):
    def __init__(self, mesh, bound_verts, model_cfg, runtime_cfg):
        super().__init__()
        scale = runtime_cfg.get("scale", 1.0)

        self.model_cfg = model_cfg
        input_channels = model_cfg.get("INPUT_CHANNEL", 5) 
    
        self.mesh = mesh
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.edge_centers = (self.vertices[self.edges[0]] + self.vertices[self.edges[1]])/2
        self.bound_verts = self.vertices[bound_verts]
        self.use_positional_encoding = model_cfg.get("USE_POSITIONAL_ENCODING", False)
        self.encoding_length = model_cfg.get("ENCODING_LENGTH", 10) 
        
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
            
        self.num_layers = runtime_cfg["num_layers"]
        self.num_innder_layer = model_cfg.get("NUM_INNER_LAYER", 3) 

        
        self.fc_w = self.make_fc_layers([self.fc1_dim, self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim,], input_channels, self.num_layers*self.num_innder_layer,)
        self.fc_a = self.make_fc_layers([self.fc1_dim, self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim ], input_channels, self.num_layers*self.num_innder_layer,) 
        
        # self.fc_w = self.make_fc_layers([ self.fc1_dim,self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim*4, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim,], input_channels, self.num_layers*self.num_innder_layer,)
        # self.fc_a = self.make_fc_layers([ self.fc1_dim,self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim*4, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim,self.fc1_dim ], input_channels, self.num_layers*self.num_innder_layer,) 
        
    
     
    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels, append_bn=False):
        fc_layers = []
        c_in = input_channels
        for k in range(0, len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=True),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.LeakyReLU(),
            ])
            c_in = fc_cfg[k]
        
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

    
    def forward(self, batch_dict):

        input_param = batch_dict['pose_param'].float()
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
        w_pred = w_pred.view(batch_size, self.num_edges, self.num_layers, self.num_innder_layer)
        
        angle_pred = self.fc_a(a_input)
        angle_pred = angle_pred.view(batch_size, self.num_bound_verts, self.num_layers, self.num_innder_layer) 
        
        for i in range(self.num_layers):
            var_pred = torch.cat((w_pred[:,:,i,:], angle_pred[:,:,i,:]), dim=1)          
            batch_dict['var_pred%d'%(i)] = var_pred

        return batch_dict
