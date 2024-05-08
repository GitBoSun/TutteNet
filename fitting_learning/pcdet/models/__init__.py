from collections import namedtuple

import numpy as np
import torch
import re

from .tutte_models import build_tutte_model

def build_network(model_cfg, cfg, dataset):
    import pcdet.models.tutte_models as tutte_models

    builder_dict = {}

    for name in dir(tutte_models):
        if name[:1].isupper():
            builder_dict[name] = build_tutte_model
            
    builder = builder_dict[model_cfg.NAME]
    model = builder(model_cfg=model_cfg, runtime_cfg=cfg, dataset=dataset)

    freezed_modules = cfg.MODEL.get('FREEZED_MODULES', None)
    if freezed_modules is not None:
        for name, param in model.named_parameters():
            for module_regex in freezed_modules:
                if re.match(module_regex, name) is not None:
                    print(f"FREEZING {name}")
                    param.requires_grad = False

    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'pred_dicts'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict, pred_dicts = model(batch_dict)
        loss = ret_dict['loss']
        # loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, pred_dicts)

    return model_func


def freeze_modules(model, freezed_modules):
    if freezed_modules is not None:
        for name, param in model.named_parameters():
            for module_regex in freezed_modules:
                if re.match(module_regex, name) is not None:
                    print(f"FREEZING {name}")
                    param.requires_grad = False
    return model

def freeze_modules_except(model, freezed_modules):
    if freezed_modules is not None:
        for name, param in model.named_parameters():
            matched=False
            for module_regex in freezed_modules:
                if re.match(module_regex, name) is not None:
                    matched = True
                    print(f"UNDO FREEZING {name}")
                    param.requires_grad = True

            if not matched:
                print(f"FREEZING {name}")
                param.requires_grad = False
    return model

