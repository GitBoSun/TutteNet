import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from timm.models.layers import trunc_normal_

from ...utils import loss_utils

class MLPHead(nn.Module):

    def __init__(self, mesh, bound_verts, runtime_cfg, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_layers = runtime_cfg["num_layers"]
        self.input_radius = model_cfg.get("INPUT_RADIUS", 0.8 )
        self.num_points = runtime_cfg["num_points"]

        # self.fc_cfg = model_cfg.get("FC_CFG", [32, 64, 128, 256, 512])
        
        # self.fc_cfg = model_cfg.get("FC_CFG", [32, 64, 256, 512, 1024, 2048, 4096])
        # self.fc_cfg = model_cfg.get("FC_CFG", [32, 64, 256,  256, 512, 512, 1024, 1024, 1024, 2048, 2048, 4096])
        # self.fc_layers = self.make_fc_layers(self.fc_cfg, 5, self.num_points*2)
        
        self.fc_cfg = model_cfg.get("FC_CFG", [400, 256, 128, 256, 400])
        self.fc_layers = self.make_fc_layers(self.fc_cfg, 400, self.num_points*2)
        
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        
        # self.init_weights() 

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels, dropout=None):
        fc_layers = []
        c_in = input_channels
        for k in range(0, len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                # nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]

        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def build_losses(self, losses_cfg):
        if not isinstance(losses_cfg['LOSS'], list):
            losses_cfg['LOSS'] = [losses_cfg['LOSS']]
        if not isinstance(losses_cfg['WEIGHT'], list):
            losses_cfg['WEIGHT'] = [losses_cfg['WEIGHT']]
        self.loss_names = losses_cfg['LOSS']
        self.losses = nn.ModuleList()
        self.loss_weight = []
        for loss, weight in zip(losses_cfg['LOSS'], losses_cfg['WEIGHT']):
            self.losses.append(
                loss_utils.LOSSES[loss](loss_cfg=losses_cfg)
            )
            self.loss_weight.append(weight)

    def get_regress_loss(self, tb_dict=None, prefix=None):

        target_points = self.forward_ret_dict['target_points']
        pred_points = self.forward_ret_dict['pred_points']

        if tb_dict is None:
            tb_dict = {}

        point_loss_cls = 0.0
        for loss_module, loss_name, loss_weight in \
                zip(self.losses, self.loss_names, self.loss_weight):
            loss_this = loss_module(pred_points, target_points)*loss_weight
            if prefix is None:
                tb_dict[loss_name] = loss_this.item()
            else:
                tb_dict[f'{prefix}/{loss_name}'] = loss_this.item()
            point_loss_cls += loss_this

        return point_loss_cls, tb_dict

    def get_loss(self, tb_dict=None, prefix=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss, tb_dict_1 = self.get_regress_loss(prefix=prefix)
        tb_dict['loss_reg'] = point_loss.item()

        return point_loss, tb_dict

    def forward(self, batch_dict):
        batch_size =  batch_dict['batch_size']
        # pred_points = batch_dict['e_param'].float()
        # for i in range(len(self.fc_layers)):
        #     pred_points = self.fc_layers[i](pred_points)
        #     print("#####", i, pred_points )
        
        # pred_points = self.fc_layers(batch_dict['e_param'].float())
        pred_points = self.fc_layers(batch_dict['target_points'].float().view(batch_size, -1))
        # print("#####", pred_points)
        pred_points = pred_points.view(batch_size, self.num_points, 2)
        ret_dict = {
            'pred_points': pred_points,
            'target_points': batch_dict['target_points'].float(),
            'batch_size': batch_size,
            'names': batch_dict['idx'],
            'e_param': batch_dict['e_param'], 
        }

        self.forward_ret_dict = ret_dict
        return batch_dict

    def get_evaluation_results(self):
        return self.forward_ret_dict

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # trunc_normal_(m.weight, std=.02)
                nn.init.uniform_(m.weight, -0.2, 0.2)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)