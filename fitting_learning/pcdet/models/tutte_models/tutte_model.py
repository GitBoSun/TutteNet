import torch
import time 

from .tutte_template import TutteTemplate

class TutteModel(TutteTemplate):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__(model_cfg=model_cfg, runtime_cfg=runtime_cfg, dataset=dataset)
        self.module_list = self.build_networks()
        self.num_pos = 0
        self.mesh_resolution = model_cfg.get("MESH_RESOLUTION", 11)
        # self.num_layers = model_cfg.get("NUM_LAYERS", 8)

    def forward(self, batch_dict,):
        
        t0 = time.time() 
        if self.backbone:
            batch_dict = self.backbone(batch_dict)
        if self.tutte_head:
            batch_dict = self.tutte_head(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            pred_dicts = self.tutte_head.get_evaluation_results()

            tb_dict['metadata/max_memory_allocated_in_GB'] = torch.cuda.max_memory_allocated() / 2**30
            disp_dict['mem'] = torch.cuda.max_memory_allocated() / 2**30
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict, pred_dicts
        else:
            loss, tb_dict, disp_dict = self.get_training_loss()
            pred_dicts = self.tutte_head.get_evaluation_results()
            return pred_dicts, loss, tb_dict

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}

        # loss = 0.0
        if self.tutte_head:
            loss, tb_dict = self.tutte_head.get_loss(tb_dict)
            # print("loss", loss)
        return loss, tb_dict, disp_dict
