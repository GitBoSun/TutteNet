import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentEncoderBig(nn.Module):
    def __init__(self, model_cfg, runtime_cfg):
        super().__init__()
        scale = runtime_cfg.get("scale", 1.0)

        self.model_cfg = model_cfg
        self.if_3d = model_cfg.get("IF_3D", False) 
        input_channels = model_cfg.get("INPUT_CHANNEL", 5) 

        num_edges = runtime_cfg["num_edges"]
        num_bound_verts = runtime_cfg["num_bound_verts"]
        self.rotate_angle = runtime_cfg["rotate_angle"]
        self.fix_bound = runtime_cfg["fix_bound"]
        self.num_innder_layer = model_cfg.get("NUM_INNER_LAYER", 3)  
        
        self.append_bn = model_cfg.get("APPEND_BN", False)  
        self.skip_connection = model_cfg.get("SKIP_CONNECTION", False) 
        self.repeat_mlp = model_cfg.get("REPEAT_MLP", False) 
        
        self.fc1_dim = model_cfg.get("FC1_DIM", 32)

        num_var = num_edges
        if not self.fix_bound:
            num_var += num_bound_verts
        if self.rotate_angle:
            num_var += 1 
        
        if self.if_3d:
            num_var = num_var*self.num_innder_layer 
            
        self.num_layers = runtime_cfg["num_layers"]

        self.fc1 = self.make_fc_layers([self.fc1_dim, self.fc1_dim, self.fc1_dim*2, self.fc1_dim*2, self.fc1_dim*4], input_channels, self.fc1_dim * 4,)
        fc_layers = []
        input_channel = self.fc1_dim*4
        self.latent_dim = input_channel 
        
       
        for i in range(self.num_layers):
            # layer1 = self.make_fc_layers([self.latent_dim*2], self.latent_dim, num_var+ self.latent_dim) 
            # fc_layers.append(layer1)
            layer1 = self.make_fc_layers([self.latent_dim*2, self.fc1_dim*4, self.fc1_dim*8], self.latent_dim, num_var,) 
            if i<self.num_layers-1:
                layer2 = self.make_fc_layers([self.latent_dim, self.fc1_dim, self.fc1_dim], self.latent_dim, self.latent_dim, append_bn=self.append_bn) 
            else:
                layer2 = None 
            fc_layers.append(nn.ModuleList([layer1, layer2]))
            
        self.fc_layers = nn.ModuleList(fc_layers)
    
        # self.init_weights() 
        
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
        if append_bn:
            fc_layers.append(nn.BatchNorm1d(output_channels)) 
            fc_layers.append(nn.LeakyReLU()) 
            
        return nn.Sequential(*fc_layers)

    def forward_skip(self, batch_dict):
        input_param = batch_dict['pose_param'].float()
        las_latent_code = self.fc1(input_param)
        last_var = None 
        
        for i in range(self.num_layers):
            layers_i = self.fc_layers[i] 
            # pred = layers_i(latent_code)
            # var_pred = pred[:, self.latent_dim:]
            # latent_code = pred[:, :self.latent_dim]
            
            var_pred = layers_i[0](las_latent_code)
            if i<self.num_layers-1:
                latent_code = layers_i[1](las_latent_code) 
                # las_latent_code = las_latent_code + latent_code 
                las_latent_code = latent_code 
                
            print("####",i, var_pred.mean().item(), torch.abs(var_pred).mean().item(), var_pred.max().item(), var_pred.min().item(), )
            if last_var is None:
                last_var =  var_pred.clone() 
            else:
                last_var = var_pred + last_var 
            batch_dict['var_pred%d'%(i)] = last_var 
            
        return batch_dict
    
    def forward(self, batch_dict):
        if self.skip_connection:
            return self.forward_skip(batch_dict)

        input_param = batch_dict['pose_param'].float()
        latent_code = self.fc1(input_param)
        
        for i in range(self.num_layers):
            if self.repeat_mlp:
                latent_code = self.layer0(latent_code)
                var_pred = self.layer1(latent_code)
                if i<self.num_layers-1:
                    latent_code = self.layer2(latent_code) 
            else: 
                layers_i = self.fc_layers[i] 
                var_pred = layers_i[0](latent_code)
                if i<self.num_layers-1:
                    latent_code = layers_i[1](latent_code) 
            # pred = layers_i(latent_code)
            # var_pred = pred[:, self.latent_dim:]
            # latent_code = pred[:, :self.latent_dim]
            
            # print("@@@@", latent_code)
            # print("####", var_pred)
            # print("####",i, var_pred.mean().item(), torch.abs(var_pred).mean().item(), var_pred.max().item(), var_pred.min().item(), )
            batch_dict['var_pred%d'%(i)] = var_pred

        return batch_dict

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.5)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)