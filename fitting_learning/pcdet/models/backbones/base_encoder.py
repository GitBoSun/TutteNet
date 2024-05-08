import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        input_channels = 5

        self.num_edges = runtime_cfg["num_edges"]
        self.num_bound_verts = runtime_cfg["num_bound_verts"]
        self.rotate_angle = runtime_cfg["rotate_angle"]
        
        self.latent_dim = model_cfg.get("LATENT_DIM", 32)
        self.fc1_dim = model_cfg.get("FC1_DIM", 64)
        self.use_latent = model_cfg.get("USE_LATENT", False) 
        self.use_bn = model_cfg.get("USE_BN", False) 
        self.append_bn = model_cfg.get("APPEND_BN", False) 
        
        # self.relu = torch.nn.ReLU() 
        
        num_var = self.num_edges + self.num_bound_verts
        if self.use_latent:
            num_var += self.latent_dim 
        if self.rotate_angle:
            num_var += 1 
            
        self.num_layers = runtime_cfg["num_layers"]

        self.fc1 = self.make_fc_layers([self.fc1_dim, self.fc1_dim*2, self.fc1_dim*4, num_var], input_channels, num_var, )
        self.fc_layers = nn.ModuleList([self.make_fc_layers([num_var], input_channels+num_var, num_var, ) for i in range(self.num_layers-1)])
        if self.append_bn:
            self.bn_layers = nn.ModuleList([nn.Sequential(*[nn.BatchNorm1d(num_var), nn.ReLU(),]) for i in range(self.num_layers-1)])
    
    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)


    def forward(self, batch_dict):
        input_param = batch_dict['e_param'].float()
        var_pred = self.fc1(input_param)
        print("******", var_pred)
        batch_dict['var_pred0'] = var_pred

        for i in range(self.num_layers-1):
            if self.append_bn:
                var_pred = self.bn_layers[i](var_pred)
            layer_input = torch.cat((input_param, var_pred), dim=1)
            var_pred = self.fc_layers[i](layer_input)
            
            weight_pred = var_pred[:self.num_edges]
            print("$$$$$$$$", weight_pred.shape, weight_pred.mean(), weight_pred.min(), weight_pred.max(), weight_pred)

            batch_dict['var_pred%d'%(i+1)] = var_pred

        return batch_dict
