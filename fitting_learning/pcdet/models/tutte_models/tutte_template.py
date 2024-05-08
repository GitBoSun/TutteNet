import os
import trimesh
import numpy as np

import torch
import torch.nn as nn

from .. import backbones, tutte_heads
from ..backbones import base_encoder
from ..tutte_heads import tutte_head

class TutteTemplate(nn.Module):
    def __init__(self, model_cfg, runtime_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.runtime_cfg = runtime_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
             'backbone', 'tutte_head',]
        self.ema = {}
        self.ema_momentum = 0.9997

        self.mesh_resolution = model_cfg.get("MESH_RESOLUTION", 11)
        self.num_layers = model_cfg.get("NUM_LAYERS", 8)
        self.num_inner_layers = model_cfg.get("NUM_INNER_LAYERS", 8) 

        self.radius = 1
        vertices, faces, bound_verts = self.build_uniform_square_graph()
        vertices = (vertices-0.5) * 2
        vertices_3d = np.zeros((vertices.shape[0], 3))
        vertices_3d[:, 0:2] = vertices
        mesh = trimesh.base.Trimesh(vertices=vertices_3d, faces=np.array(faces))
        edges = np.array(mesh.edges).transpose()
        edges = np.concatenate((edges, edges[[1,0],:]), axis=1)
        edges = np.unique(edges, axis=1)

        mesh_input = {}
        mesh_input['vertices'] = torch.from_numpy(vertices).float().cuda()
        mesh_input['edges'] = torch.from_numpy(edges).long().cuda()
        mesh_input['faces'] = faces

        self.num_edges = edges.shape[1]
        self.num_bound_verts = bound_verts.shape[0]
        self.bound_verts = bound_verts
        self.mesh_input = mesh_input
        
        self.rotate_angle = model_cfg.get("ROTATE_ANGLE", False) 
        self.fix_bound = model_cfg.get("FIX_BOUND", False) 
        self.predict_normal = model_cfg.get("PREDICT_NORMAL", False)
        self.use_shape = model_cfg.get("USE_SHAPE", False)
        self.use_depth = model_cfg.get("USE_DEPTH", False)
        self.num_depth = model_cfg.get("NUM_DEPTH", 1)
        
    def update_ema(self):
        for name, v in self.named_parameters():
            ema_v = self.ema.get(name, None)
            if ema_v is None:
                self.ema[name] = v.clone()
            else:
                self.ema[name] = ema_v * self.ema_momentum + v * (1-self.ema_momentum)

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_layers': self.num_layers,
            'num_inner_layers': self.num_inner_layers, 
            'num_edges': self.num_edges,
            'num_bound_verts': self.num_bound_verts,
            'rotate_angle': self.rotate_angle, 
            'fix_bound': self.fix_bound, 
            'predict_normal': self.predict_normal,
            'use_shape': self.use_shape, 
            'use_depth': self.use_depth, 
            'num_depth': self.num_depth, 
            'add_gender': self.dataset.add_gender, 
        }
        if hasattr(self.dataset, "num_points"):
            model_info_dict.update({'num_points': self.dataset.num_points,})
        elif hasattr(self.dataset, "num_points1"):
            model_info_dict.update({'num_points1': self.dataset.num_points1,})
            model_info_dict.update({'num_points2': self.dataset.num_points2,})
        
        if hasattr(self.dataset, "temp_vert"):
            model_info_dict.update({'temp_vert': self.dataset.temp_vert,
                                    'fps_ids': self.dataset.fps_ids,
                                    'temp_face': self.dataset.temp_face,
                                    
                                    })
            
        model_info_dict.update(self.dataset.runtime_cfg)

        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone(self, model_info_dict):
        if self.model_cfg.get('BACKBONE', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones.__all__[self.model_cfg.BACKBONE.NAME](
            model_cfg=self.model_cfg.BACKBONE,
            runtime_cfg=model_info_dict,
            mesh = self.mesh_input,
            bound_verts = self.bound_verts,
        )
        model_info_dict['module_list'].append(backbone_3d_module)

        return backbone_3d_module, model_info_dict

    def build_tutte_head(self, model_info_dict):
        if self.model_cfg.get('TUTTE_HEAD', None) is None:
            return None, model_info_dict

        # num_point_features = model_info_dict['num_point_features']
        # model_info_dict['input_channels'] = num_point_features
        point_head_module = tutte_heads.__all__[self.model_cfg.TUTTE_HEAD.NAME](
            model_cfg=self.model_cfg.TUTTE_HEAD,
            runtime_cfg=model_info_dict,
            mesh = self.mesh_input,
            bound_verts = self.bound_verts,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def build_uniform_square_graph(self, ):
        N = self.mesh_resolution
        grid_interval = 1/float(N-1)
        bound_verts = []
        vertices = np.zeros((N*N+(N-1)*(N-1), 2))
        faces = []

        for i in range(N):
            for j in range(N):
                vertices[i*N+j] = np.array([j*grid_interval, 1-i*grid_interval])

                if i<N-1 and j<N-1:
                    vertices[i*(N-1)+j+N*N] = np.array([j*grid_interval+grid_interval/2, 1-i*grid_interval-grid_interval/2])

        for i in range(N-1):
            for j in range(N-1):
                faces.append([i*(N-1)+j + N*N, i*N+j, i*N+j+1])
                faces.append([i*(N-1)+j + N*N, i*N+j+1, (i+1)*N+j+1])
                faces.append([i*(N-1)+j + N*N, (i+1)*N+j+1, (i+1)*N+j])
                faces.append([i*(N-1)+j + N*N, (i+1)*N+j, i*N+j])

        # get boundary vertices
        j = N-1
        for i in range(N//2-1, 0, -1):
            bound_verts.append(i*N+j)
        i = 0
        for j in range(N-1, 0, -1):
            bound_verts.append(i*N+j)
        j = 0
        for i in range(N-1):
            bound_verts.append(i*N+j)
        i = N-1
        for j in range(N-1):
            bound_verts.append(i*N+j)
        j = N-1
        for i in range(N-1, N//2-1, -1):
            bound_verts.append(i*N+j)

        return vertices, np.array(faces, np.int32), np.array(bound_verts, np.int32)

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        update_model_state = {}
        for key, val in model_state_disk.items():

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def _load_and_ema_state_dict(self, model_state_disk, momentum, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        update_model_state = {}
        for key, val in model_state_disk.items():

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val * (1-momentum) + state_dict[key] * momentum
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, ema=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if ema:
            model_state_ema = model_state_disk.pop('ema')
            model_state_disk.update(model_state_ema)

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s in state dict: %s, no key matching' % (key, str(state_dict[key].shape)))
            else:
                logger.info('Updated weight %s in state dict from file: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_ema_params_from_files(self, filenames, logger, momentum, to_cpu=False):
        for i, filename in enumerate(filenames):
            if not os.path.isfile(filename):
                raise FileNotFoundError

            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            loc_type = torch.device('cpu') if to_cpu else None
            checkpoint = torch.load(filename, map_location=loc_type)
            model_state_disk = checkpoint['model_state']

            version = checkpoint.get("version", None)
            if version is not None:
                logger.info('==> Checkpoint trained from version: %s' % version)

            if i == 0:
                state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
            else:
                state_dict, update_model_state = self._load_and_ema_state_dict(model_state_disk, momentum, strict=False)

            for key in state_dict:
                if key not in update_model_state:
                    logger.info('Not updated weight %s in state dict: %s, no key matching' % (key, str(state_dict[key].shape)))
                else:
                    logger.info('Updated weight %s in state dict from file: %s' % (key, str(state_dict[key].shape)))

            logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
