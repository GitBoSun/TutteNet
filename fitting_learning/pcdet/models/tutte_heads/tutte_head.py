import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter
from torch_sparse_solve import solve

import random
import trimesh
import numpy as np
import numpy.linalg
from scipy.spatial import Delaunay

from ...utils import loss_utils

class TutteLayer(nn.Module):
    def __init__(self, mesh, lap_dict, model_cfg, runtime_cfg, radius=1,):
        super(TutteLayer, self).__init__()
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.radius = radius
        self.epsilon = model_cfg.get("EPSILON", 0.0001)
        self.epsilon_angle = model_cfg.get("EPSILON_ANGLE", 0.0001)
        self.circle_map = False
        
        self.use_sigmoid = model_cfg.get("USE_SIGMOID", True)
        self.use_normalize = model_cfg.get("USE_NORMALIZE", False)
        self.use_plain_scale = model_cfg.get("USE_PLAIN_SCALE", False)
        self.divide_max = model_cfg.get("DIVIDE_MAX", False)
        self.gather_middle_results = model_cfg.get("GATHER_MIDDLE_RESULTS", False)
        self.area_loss_weight = model_cfg.get("AREA_LOSS_WEIGHT", False)
        self.rotate_angle = runtime_cfg["rotate_angle"]
        self.fix_bound = runtime_cfg["fix_bound"]
        self.gau_normalize = model_cfg.get("GAU_NORMALIZE", False)
        self.w_scale = model_cfg.get("W_SCALE", 100.0)
        self.angle_scale = model_cfg.get("ANGLE_SCALE", 100.0)
        self.remove_rotate = model_cfg.get("REMOVE_ROTATE", False)
        self.inverse_w = model_cfg.get("INVERSE_W", False) 
        
        self.angle_sigmoid = model_cfg.get("ANGLE_SIGMOID", True) 
        self.angle_abs = model_cfg.get("ANGLE_ABS", False) 
        
        self.num_edges = self.edges.shape[1]
        self.lap_dict = lap_dict

        self.interior_verts = self.lap_dict['interior_verts']
        self.inter_vert_mapping = self.lap_dict['inter_vert_mapping']
        self.bound_verts = self.lap_dict['bound_verts']
        self.bound_ids = self.lap_dict['bound_ids']
        self.bound_b_val_ids = self.lap_dict['bound_b_val_ids']
        self.interior_ids = self.lap_dict['interior_ids']
        self.lap_size = self.lap_dict['lap_index'].shape[1]
        
        self.tri = Delaunay(self.vertices.cpu().numpy())
        self.tri.simplices = np.array(self.faces, np.int32)

    def forward(self, input_points, var_pred):

        W_var = var_pred[:, :self.num_edges]
        if self.fix_bound:
            angle_var = None 
        elif not self.rotate_angle:
            angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)]
        else:
            angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)+1]
        
        # if self.use_normalize:
        #     W_var = F.normalize(W_var)
        #     angle_var = F.normalize(angle_var)
        # elif self.divide_max:
        #     W_var = W_var/(W_var.max(1)[0].unsqueeze(1) + 1) 
        #     angle_var = angle_var/(angle_var.max(1)[0].unsqueeze(1) + 1) 
        # # elif self.gau_normalize:
        # elif 0: 
        #     mean_w = W_var.mean(1).unsqueeze(1) 
        #     var_w = torch.std(W_var, dim=1).unsqueeze(1) 
        #     W_var = (W_var - mean_w )/var_w 
            
        #     mean_a = angle_var.mean(1).unsqueeze(1) 
        #     var_a = torch.std(angle_var, dim=1).unsqueeze(1) 
        #     angle_var = (angle_var - mean_a )/var_a

        angle_var = angle_var / self.angle_scale
            
        # print("********", W_var)
        if self.angle_abs:
            angle_var = torch.abs(angle_var)
        elif self.angle_sigmoid:
            angle_var = torch.sigmoid(angle_var) * (1-2*self.epsilon_angle) + self.epsilon_angle
            # angle_var = torch.sigmoid(angle_var) * (1-2*self.epsilon) + self.epsilon
        # W_var = torch.abs(W_var)
        if self.inverse_w:
            W_var = torch.abs(W_var)
            W_var = 1/(W_var + 1)
            # W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
            # W_var = 2* (W_var - 0.5 )

        elif self.use_sigmoid:
            W_var = W_var/self.w_scale
            W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
        
        # print(W_var.min(), W_var.max())
       
        # if self.use_sigmoid:
        #     angle_var = torch.sigmoid(angle_var) * (1-2*self.epsilon) + self.epsilon
        #     W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
        #     # angle_var = torch.sigmoid(torch.clamp(angle_var, -10, 10)) * (1-2*self.epsilon) + self.epsilon
        #     # W_var = torch.sigmoid(torch.clamp(W_var, -10, 10))
        # else:
        #     angle_var = torch.abs(angle_var)
        #     W_var = torch.abs(W_var)
        #     # angle_var = torch.clamp(angle_var, 0.05, 0.9)
        #     # W_var = torch.clamp(W_var, 0.05, 1)
        
        # print("#######", W_var)
        # print(angle_var)
        input_points_np = input_points.clone().detach().cpu().numpy()
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        input_points_np = np.reshape(input_points_np, (batch_size*N_points, 2))
        face_ids = Delaunay.find_simplex(self.tri, input_points_np.astype(np.float64), bruteforce=True)
        input_nodes = self.faces[face_ids]
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)

        input_areas = self.get_areas(input_points, input_nodes) # [b, N, 3]
        new_vertices = self.tutte_embedding_sparse(W_var, angle_var)
        pred_points = self.get_tutte_from_triangle(new_vertices, input_nodes, input_areas)
        
        d_faces = np.reshape(self.faces[face_ids], (batch_size, N_points, 3)).astype(np.int32)
        distortions = self.compute_distortion(d_faces, self.vertices, new_vertices)
        return_dict = {'new_vertices': new_vertices, 'distortion': distortions}
        return pred_points, return_dict

    def tutte_embedding_sparse(self, W_var, angle_var):
        """
        Args:
            W_var: [b, n_edge]
            angle_var: [b, n_bound]
        """
        n_vert = self.vertices.shape[0]
        batch_size = W_var.shape[0]
        
        if angle_var is None:
            angle_init = 2 * torch.pi * torch.arange(len(self.bound_verts)).float().cuda()/len(self.bound_verts)
            angle_init = angle_init.unsqueeze(0).repeat(batch_size, 1)
            
        elif self.rotate_angle:
            tmp_angle =  angle_var[:, 1:].clone()
            new_angle = tmp_angle/tmp_angle.sum(1).unsqueeze(1)
            angle_init = torch.cumsum(new_angle, dim=1) + angle_var[:, 0].unsqueeze(1)
            angle_init = angle_init * 2 * torch.pi
            # angle_var = torch.cat((angle_var[:, 0].unsqueeze(1), tmp_angle), dim=1)
            # angle_var[:, 1:] = angle_var[:, 1:]/angle_var[:, 1:].sum(1).unsqueeze(1)
        else:
            new_angle = angle_var/angle_var.sum(1).unsqueeze(1)
            angle_init = torch.cumsum(new_angle, dim=1) * 2 * torch.pi # [b, n_bound]

        angle_init = angle_init.view(-1, )
        bound_pos = torch.zeros((batch_size * len(self.bound_verts), 2)).cuda() # [b, n_bound, 2]

        if self.circle_map:
            bound_pos[:, 0] = self.radius * torch.cos(angle_init)
            bound_pos[:, 1] = self.radius * torch.sin(angle_init)
        else:
            mask1 = (angle_init > 7*torch.pi/4) | (angle_init <= 1*torch.pi/4)
            mask2 = (angle_init > torch.pi/4) & (angle_init <= 3*torch.pi/4)
            mask3 = (angle_init > 3*torch.pi/4) & (angle_init <= 5*torch.pi/4)
            mask4 = (angle_init > 5*torch.pi/4) & (angle_init <= 7*torch.pi/4)

            bound_pos[mask1, 0] = 1.0
            bound_pos[mask1, 1] = torch.tan(angle_init[mask1])

            bound_pos[mask2, 0] = 1/torch.tan(angle_init[mask2])
            bound_pos[mask2, 1] = 1.0

            bound_pos[mask3, 0] = -1.0
            bound_pos[mask3, 1] = -torch.tan(angle_init[mask3])

            bound_pos[mask4, 0] = -1/torch.tan(angle_init[mask4])
            bound_pos[mask4, 1] = -1.0

        bound_pos = bound_pos.view(batch_size, len(self.bound_verts), 2)

        lap_values = torch.zeros((batch_size, self.lap_size, )).cuda()
        for bi in range(batch_size):
            lap_index, lap_value = torch_geometric.utils.get_laplacian(self.edges, W_var[bi])
            lap_values[bi] = lap_value

        # self.bound_b_val_ids: [N_vert, N_bound_vert]
        # b: [b, n_vert, n_bound, 2]
        b = - ((self.bound_b_val_ids.unsqueeze(0).unsqueeze(-1)>=0) * lap_values[:, self.bound_b_val_ids].unsqueeze(-1) * \
                bound_pos.unsqueeze(1)).sum(2) # [b, n_vert, 2]

        lap_index = lap_index[:, self.interior_ids]
        lap_values = lap_values[:, self.interior_ids]
        lap_values = lap_values.view(-1)

        lap_index[0,] = self.inter_vert_mapping[lap_index[0,]]
        lap_index[1,] = self.inter_vert_mapping[lap_index[1,]]

        batch_dim = torch.zeros((batch_size * lap_index.shape[1])).long().cuda()
        for bi in range(batch_size):
            batch_dim[bi*lap_index.shape[1]: (bi+1)*lap_index.shape[1]] = bi

        # lap_index = torch.cat((torch.zeros(1, lap_index.shape[1]), lap_index.cpu()), dim=0).long()
        lap_indices = torch.cat((batch_dim.unsqueeze(0), lap_index.repeat(1, batch_size)), dim=0).long()

        A = torch.sparse_coo_tensor(lap_indices.cpu(), lap_values.double().cpu(), (batch_size, len(self.interior_verts), len(self.interior_verts)))
        b = b[:, self.interior_verts].double()
        x =  solve(A, b.cpu())

        out_pos = torch.zeros(batch_size, n_vert, 2).cuda()
        out_pos[:, self.bound_verts] = bound_pos.float()
        out_pos[:, self.interior_verts] = x.float().cuda()
        return out_pos

    def get_areas(self, points, tri_nodes,):
        """

        Args:
            points: [b, N, 3]
            tri_nodes: [b, N, 3]
        """

        loc = points
        init_A = self.vertices[tri_nodes[:,:,0].astype(np.int32), :]
        init_B = self.vertices[tri_nodes[:,:,1].astype(np.int32), :]
        init_C = self.vertices[tri_nodes[:,:,2].astype(np.int32), :]
        area_A = torch.abs((loc-init_B)[:,:,0]*(loc-init_C)[:,:,1] - (loc-init_B)[:,:,1]*(loc-init_C)[:,:,0])/2
        area_B = torch.abs((loc-init_A)[:,:,0]*(loc-init_C)[:,:,1] - (loc-init_A)[:,:,1]*(loc-init_C)[:,:,0])/2
        area_C = torch.abs((loc-init_A)[:,:,0]*(loc-init_B)[:,:,1] - (loc-init_A)[:,:,1]*(loc-init_B)[:,:,0])/2
        areas = torch.zeros(tri_nodes.shape).cuda()
        areas[:,:, 0] = area_A
        areas[:,:, 1] = area_B
        areas[:,:, 2] = area_C

        return areas

    def get_tutte_from_triangle(self, pos, tri_nodes, areas):
        """
        Args:
            pos: [b, N_vert, 3]
            tri_nodes: [b, N, 3]
            areas: [b, N, 3]

        """
        batch_size, n_points, _ = areas.shape
        n_vert = pos.shape[1]
        pos = pos.view(batch_size * n_vert, 2)

        areas = areas.view(batch_size * n_points, -1)
        for i in range(batch_size):
            tri_nodes[i] += n_vert * i
        tri_nodes = np.reshape(tri_nodes, (batch_size * n_points, 3))

        area_A = areas[:, 0].unsqueeze(1)
        area_B = areas[:, 1].unsqueeze(1)
        area_C = areas[:, 2].unsqueeze(1)
        total_area = area_A + area_B + area_C
        new_A = pos[tri_nodes[:,0].astype(np.int32)] # [b, N, 2]
        new_B = pos[tri_nodes[:,1].astype(np.int32)]
        new_C = pos[tri_nodes[:,2].astype(np.int32)]
        pred_points = (new_A * area_A + new_B * area_B + new_C * area_C)/total_area
        pred_points = pred_points.view(batch_size, n_points, 2)
        # import pdb; pdb.set_trace()
        return pred_points
    
    def compute_distortion(self, faces, vert1, vert2 ):
        """
        faces: [B, n_points, 3]
        vert1: [N, 3]
        vert2: [B, N, 3]
        """
        batch_size, n_points, _ = faces.shape 
        n_vert = vert1.shape[0]
        
        flattern_faces = np.reshape(faces, (batch_size*n_points*3))
        or_vert = vert1[flattern_faces].view(batch_size*n_points, 3,2)
        new_faces = faces.copy()
        for i in range(batch_size):
            new_faces[i] += i * n_vert 
        
        new_faces = np.reshape(new_faces, (batch_size*n_points*3)) 
        flatten_vert2 = vert2.view(batch_size*n_vert, 2)
        new_vert = flatten_vert2[new_faces].view(batch_size*n_points, 3,2)
        A = torch.zeros(batch_size*n_points, 6, 6).cuda()
        b = torch.zeros(batch_size*n_points, 6).cuda()
        A[:, 0, 0:2] = or_vert[:, 0,:]
        A[:, 0, 4] = 1 
        A[:, 1, 2:4] = or_vert[:, 0,:]
        A[:, 1, 5] = 1 
        
        A[:, 2, 0:2] = or_vert[:, 1,:]
        A[:, 2, 4] = 1 
        A[:, 3, 2:4] = or_vert[:, 1,:]
        A[:, 3, 5] = 1 
        
        A[:, 4, 0:2] = or_vert[:, 2,:]
        A[:, 4, 4] = 1 
        A[:, 5, 2:4] = or_vert[:, 2,:]
        A[:, 5, 5] = 1 
        
        b[:, 0:2] = new_vert[:, 0,:]
        b[:, 2:4] = new_vert[:, 1,:]
        b[:, 4:6] = new_vert[:, 2,:]
        
        distortion = torch.linalg.solve(A, b) # [b*n, 6]
        distortion = distortion[:, 0:4].view(batch_size* n_points, 2,2)
        return distortion 

class TutteHead(nn.Module):

    def __init__(self, mesh, bound_verts, runtime_cfg, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_layers = runtime_cfg["num_layers"]
        
        self.gather_middle_results = model_cfg.get("GATHER_MIDDLE_RESULTS", False)
        
        self.w_scale = model_cfg.get("W_SCALE", 100.0)
        self.angle_scale = model_cfg.get("ANGLE_SCALE", 100.0)
        
        self.area_loss_weight = model_cfg.get("AREA_LOSS_WEIGHT", False)
        self.add_lap_loss = model_cfg.get("ADD_LAP_LOSS", False)
        self.lap_loss_weight = model_cfg.get("LAP_LOSS_WEIGHT", 0)
        self.layer_distortion_weight = model_cfg.get("LAYER_DISTORTION_WEIGHT", 0)
        self.add_layer_distortion_loss = model_cfg.get("ADD_LAYER_DISTORTION_LOSS", False)
                
        self.mesh = mesh
        self.bound_verts = bound_verts
        self.input_radius = model_cfg.get("INPUT_RADIUS", 0.8 )
        self.num_points = runtime_cfg["num_points"]
        
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.interior_verts = []
        for v in range(len(self.vertices)):
            if not v in self.bound_verts:
                self.interior_verts.append(v)

        self.inter_vert_mapping = torch.zeros(len(self.vertices)).long().cuda()
        self.inter_vert_mapping[self.interior_verts] = torch.arange(len(self.interior_verts)).cuda()

        n_edges = self.edges.shape[1]
        W_var = torch.ones(n_edges).cuda()
        self.tri = Delaunay(self.vertices.cpu().numpy())
        self.tri.simplices = np.array(self.faces, np.int32)
        self.lap_index, lap_value = torch_geometric.utils.get_laplacian(self.edges, W_var)
        self.lap_size = lap_value.shape[0]
        
        self.bound_ids = []
        for vb in self.bound_verts:
            ids_i = torch.nonzero(self.lap_index[0, :]==vb, as_tuple=True)[0]
            self.bound_ids += ids_i
            ids_i = torch.nonzero(self.lap_index[1, :]==vb, as_tuple=True)[0]
            self.bound_ids += ids_i

        self.bound_b_val_ids = -1*torch.ones(len(self.vertices), len(self.bound_verts)).long().cuda()
        for bi, vb in enumerate(self.bound_verts):
            ids_b = torch.nonzero(self.lap_index[1, :]==vb , as_tuple=True)[0]
            self.bound_b_val_ids[self.lap_index[0][ids_b], bi] = ids_b

        self.interior_ids = []
        for i in range(self.lap_index.shape[1]):
            if not i in self.bound_ids:
                self.interior_ids.append(i)
        
        if self.add_lap_loss:
            self.diag_ids = torch.nonzero(self.lap_index[0]==self.lap_index[1], as_tuple=True)[0]
    
        self.lap_dict = {'interior_verts': self.interior_verts, 'inter_vert_mapping':self.inter_vert_mapping, 'bound_verts':self.bound_verts, \
            'bound_ids': self.bound_ids, 'bound_b_val_ids': self.bound_b_val_ids, 'interior_ids': self.interior_ids, 'lap_index': self.lap_index}
        self.tutte_layers = nn.ModuleList([TutteLayer(mesh, self.lap_dict,  model_cfg, runtime_cfg, ) \
                for i in range(self.num_layers)])
        
        self.build_losses(self.model_cfg.LOSS_CONFIG)

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

        if self.area_loss_weight:
            e_param = self.forward_ret_dict['e_param']
            area_weights = e_param[:, 0] * e_param[:, 1]
            area_weights = 1/area_weights
        else:
            area_weights = None

        if tb_dict is None:
            tb_dict = {}

        point_loss_cls = 0.0
        for loss_module, loss_name, loss_weight in \
                zip(self.losses, self.loss_names, self.loss_weight):
            loss_this = loss_module(pred_points, target_points, weights=area_weights)*loss_weight
            if prefix is None:
                tb_dict[loss_name] = loss_this.item()
            else:
                tb_dict[f'{prefix}/{loss_name}'] = loss_this.item()
            point_loss_cls += loss_this

        return point_loss_cls, tb_dict

    def get_laplacian_loss(self,):
        edges = self.mesh['edges']
        lap_loss = 0 
        for i in range(self.num_layers):
            var_pred = self.forward_ret_dict['var_pred%d'%(i)]
            for j in range(var_pred.shape[0]):
                w_var = var_pred[j, :edges.shape[1]]
                _, lap_value = torch_geometric.utils.get_laplacian(self.edges, w_var)
                diag_values = lap_value[self.diag_ids] 
                lap_loss += torch.square(diag_values - 1).mean() 
        lap_loss = lap_loss / (self.num_layers *var_pred.shape[0])
        return lap_loss
    
    def get_layer_distortion_loss(self,):
        d_loss = 0.0 
        for i in range(self.num_layers):
            distortion = self.forward_ret_dict['distortion%d'%(i)] 
            d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
            loss = torch.square(d - torch.eye(2).cuda().unsqueeze(0).repeat(d.shape[0], 1,1)).mean() 
            d_loss += loss 
        d_loss = d_loss / self.num_layers 
        return d_loss 
    
    def get_loss(self, tb_dict=None, prefix=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss, tb_dict_1 = self.get_regress_loss(prefix=prefix)
        tb_dict['loss_reg'] = point_loss.item()
        if self.add_lap_loss:
            lap_loss = self.get_laplacian_loss()
            print("l_lossssss", point_loss.item(), lap_loss.item())
            point_loss += self.lap_loss_weight * lap_loss 
            
        if self.add_layer_distortion_loss:
            distortion_loss = self.get_layer_distortion_loss()
            print("d_lossssss", point_loss.item(), distortion_loss.item())
            point_loss = point_loss + self.layer_distortion_weight * distortion_loss 
            
            
        # loss_w = 0.0 
        # for i in range(self.num_layers):
        #     var_pred = self.forward_ret_dict['var_pred%d'%(i)]
        #     w_pred = var_pred[:, :1280]
        #     loss_w += ((w_pred**2).mean(1)).mean()
        # print("lossssss", point_loss.item(), loss_w.item())
        # point_loss += 0.00001 * loss_w 
        # point_loss = loss_w 
        print("point_loss", point_loss)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        batch_size =  batch_dict['batch_size']

        theta = 2*torch.pi*torch.arange(self.num_points).float().cuda()/self.num_points
        input_points = self.input_radius * torch.concat((torch.cos(theta).unsqueeze(1), torch.sin(theta).unsqueeze(1)), dim=1)
        input_points = input_points.unsqueeze(0).repeat(batch_size, 1,1)
        input_points.requires_grad = True 
        
        ret_dict = {
            'e_param': batch_dict['e_param'],
            'target_points': batch_dict['target_points'].float(),
            'simplices': self.tutte_layers[0].tri.simplices,
            'vertices': self.mesh['vertices'],
            'batch_size': input_points.shape[0],
            'names': batch_dict['idx'],
            'input_points': input_points,
        }  

        pred_points = input_points.clone()
        total_distortion = None 
        
        for i in range(self.num_layers):
            var_pred = batch_dict['var_pred%d'%(i)]
            # pred_points = torch.zeros(input_points.shape).cuda()
            # new_vertices = torch.zeros(var_pred.shape[0], 100, 2).cuda()
            pred_points, return_dict = self.tutte_layers[i](pred_points, var_pred)
            new_vertices = return_dict['new_vertices']
            if 'distortion' in return_dict.keys():
                distortion = return_dict['distortion']
                if total_distortion is None:
                    total_distortion = distortion 
                else: 
                    total_distortion = torch.matmul(distortion, total_distortion)
                batch_dict['distortion%d'%(i)] = distortion.clone()
                ret_dict.update({'distortion%d'%(i): batch_dict['distortion%d'%(i)]})
                
            batch_dict['pred_points%d'%(i+1)] = pred_points.clone()
            batch_dict['new_vertices%d'%(i+1)] = new_vertices.clone()
            ret_dict.update({'var_pred%d'%(i): batch_dict['var_pred%d'%(i)]})
            
            if self.gather_middle_results:
                ret_dict.update({
                    'pred_points%d'%(i+1): pred_points.clone(),
                    'new_vertices%d'%(i+1): new_vertices.clone(),
                })
        
        total_distortion = total_distortion.view(batch_size, self.num_points, 2,2)
        target_points = batch_dict['target_points'].float()  
        batch_dict['total_distortion'] = total_distortion
        
        ret_dict.update({
            'pred_points': pred_points.clone(),
            'new_vertices': new_vertices.clone(),
            'target_points': target_points.clone(), 
            'total_distortion': total_distortion.clone() 
        })
        
        # def exp_func(x):
        #     pred_points = x.clone()
        #     for i in range(self.num_layers):
        #         var_pred = batch_dict['var_pred%d'%(i)]
        #         pred_points, _ = self.tutte_layers[i](pred_points, var_pred)
        #     return pred_points 
        # jacob = torch.autograd.functional.jacobian(exp_func, input_points)
        # ids = range(self.num_points)
        # ids1 = range(batch_size)
        # jacob = jacob[:, ids, :, :, ids, :][:, ids1, :, ids1, :]
        # for i in range(batch_size):
        #     for j in range(self.num_points):
        #         diff = torch.norm(jacob[i,j] - total_distortion[i,j])
        #         if diff > 1e-4:
        #             print(i,j,diff)
        # import pdb; pdb.set_trace()
        
        self.forward_ret_dict = ret_dict
        return batch_dict
    
    def get_evaluation_results(self):
        return self.forward_ret_dict
