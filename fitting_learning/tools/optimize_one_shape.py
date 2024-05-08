import time
import os 
import cv2 
import random
import trimesh
import numpy as np
import numpy.linalg
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
from torch_cluster import fps

import torch
import torch.nn as nn 
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import torch_geometric 
from torch_scatter import scatter
from torch_sparse_solve import solve
from cholespy import CholeskySolverF, MatrixType

# from pcdet.models.tutte_heads.tutte_h
# ead_3d import TutteLayer, TutteTriplane

def build_uniform_square_graph(N):
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

def distortion_loss(distortion, dim=3):
    # [N, 2,2]
    d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
    loss = torch.square(d - torch.eye(dim).cuda().unsqueeze(0).repeat(d.shape[0], 1,1)).mean()  
    return loss  

class TutteLayer(nn.Module):
    def __init__(self, mesh, lap_dict, radius=1, use_sigmoid=True, circle_map=False):
        super(TutteLayer, self).__init__()
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.radius = radius
        self.bound_verts = bound_verts 
        self.use_sigmoid = use_sigmoid 
        self.circle_map = circle_map
        self.batch_size = 1 
        
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
        
       
        n_edges = self.edges.shape[1]
        W_var = torch.empty(self.batch_size, n_edges).cuda()
        torch.nn.init.uniform_(W_var)
        self.W_var = nn.Parameter(W_var)
        
        angle_var = torch.empty(self.batch_size, len(bound_verts)).cuda()
        torch.nn.init.uniform_(angle_var)
        self.angle_var = nn.Parameter(angle_var) 

    def forward(self, input_points, ):
        
        # W_var = var_pred[:, :len(self.edges)]
        # angle_var = var_pred[:, len(self.edges):]
        
        input_points_np = input_points.detach().cpu().numpy() 
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        input_points_np = np.reshape(input_points_np, (batch_size*N_points, 2))
        face_ids = Delaunay.find_simplex(self.tri, input_points_np.astype(np.float64), bruteforce=True) 
        input_nodes = self.faces[face_ids] 
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)
        
#         input_nodes = np.zeros((batch_size, N_points, 3), np.int32) # [b, N]
        
#         for bi in range(batch_size):
#             face_ids = Delaunay.find_simplex(self.tri, input_points_np[bi].astype(np.float64), bruteforce=True)
#             input_nodes[bi] = self.faces[face_ids] 
            
        input_areas = self.get_areas(input_points, input_nodes) # [b, N, 3]
        new_vertices = self.tutte_embedding_sparse() 
        # print("new_vertices", new_vertices.shape, (new_vertices[0] - new_vertices[1]).max(0))
        pred_points = self.get_tutte_from_triangle(new_vertices, input_nodes, input_areas)
        
        if 1:
            d_faces = np.reshape(self.faces[face_ids], (batch_size, N_points, 3)).astype(np.int32)
            distortions = self.compute_distortion(d_faces, self.vertices, new_vertices)
            # print(torch.autograd.grad(pred_points.sum(), input_points[0][0], retain_graph=True))
            # import pdb; pdb.set_trace()
            return pred_points, new_vertices, distortions 
        
        return pred_points, new_vertices 

    def tutte_embedding_sparse(self, ):
        """
        Args:
            W_var: [b, n_edge]
            angle_var: [b, n_bound]
        """
        n_vert = self.vertices.shape[0]
        W_var = torch.sigmoid(self.W_var,)*(1-0.4) + 0.2
        
        # if self.use_sigmoid:
        #     W_var = torch.sigmoid(torch.clamp(self.W_var, -10, 10))
        # else:
        #     W_var = torch.clamp(self.W_var, 0.05, 1) 
        
        batch_size = W_var.shape[0]
        angle_var = torch.sigmoid(self.angle_var)*(1-0.4) + 0.2
        
        # if self.use_sigmoid:
        #     angle_var = torch.sigmoid(torch.clamp(self.angle_var, -10, 10)) 
        # else:
        #     angle_var = torch.clamp(self.angle_var, 0.05, 1) 
        
#         if 1:
#             tmp_angle =  angle_var[:, 1:].clone() 
#             tmp_angle = tmp_angle/tmp_angle.sum(1).unsqueeze(1)               
#             angle_var = torch.cat((angle_var[:, 0].unsqueeze(1), tmp_angle), dim=1)
            
#             # angle_var[:, 1:] = angle_var[:, 1:]/angle_var[:, 1:].sum(1).unsqueeze(1)      
#         else:    
#             angle_var = angle_var/angle_var.sum(1).unsqueeze(1)
#         import pdb; pdb.set_trace()

        angle_var = angle_var/angle_var.sum(1).unsqueeze(1)
        angle_init = torch.cumsum(angle_var, dim=1) * 2 * torch.pi # [b, n_bound]
        angle_init = angle_init.view(-1, )
        bound_pos = torch.zeros((batch_size * angle_var.shape[1], 2)).cuda() # [b, n_bound, 2]
        
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
            
        bound_pos = bound_pos.view(batch_size, angle_var.shape[1], 2)
        
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
        
        A = torch.sparse_coo_tensor(lap_indices, lap_values.double(), (batch_size, len(self.interior_verts), len(self.interior_verts)))
        b = b[:, self.interior_verts].double()        
        x =  solve(A.cpu(), b.cpu()).cuda()
        
        out_pos = torch.zeros(batch_size, n_vert, 2).cuda() 
        out_pos[:, self.bound_verts] = bound_pos.float()
        out_pos[:, self.interior_verts] = x.float()
        
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
        offset = 0 
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
        # print('ggg', pred_points[0]-pred_points[1])
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
        # import pdb; pdb.set_trace()
        return distortion 
    
class TutteTriplane(nn.Module):
    def __init__(self, mesh, lap_dict, radius=1, prefix=''):
        super(TutteTriplane, self).__init__()
        self.tutte_layer1 = TutteLayer(mesh, lap_dict, radius)
        self.tutte_layer2 = TutteLayer(mesh, lap_dict, radius)
        self.tutte_layer3 = TutteLayer(mesh, lap_dict, radius)
        self.prefix = prefix 
    
    def forward(self, input_points):
        prefix = self.prefix
        new_points = input_points
    
        cur_points1 = new_points[:,:,0:2].clone() 
        pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1)
        new_points1 = torch.cat((pred_points1, new_points[:,:,2].unsqueeze(2)), dim=2)
        
        cur_points2 = torch.cat((new_points1[:,:,0].unsqueeze(2), new_points1[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2)
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points1[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2) 
        
        cur_points3 = new_points2[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3)
        new_points3 = torch.cat((new_points2[:,:,0].unsqueeze(2), pred_points3), dim=2)
        
        total_distortion = self.compute_distortion(distortions1, distortions2,distortions3)
        
        return_dict = {} 
        return_dict[prefix+'pred_points1'] = new_points1 
        return_dict[prefix+'new_01_vertices1'] = new_vertices1
        return_dict[prefix+'pred_01_points1'] = pred_points1
        return_dict[prefix+'distortion1'] = distortions1 
        
        return_dict[prefix+'pred_points2'] = new_points2
        return_dict[prefix+'new_02_vertices2'] = new_vertices2 
        return_dict[prefix+'pred_02_points2'] = pred_points2
        return_dict[prefix+'distortion2'] = distortions2
        
        return_dict[prefix+'pred_points3'] = new_points3
        return_dict[prefix+'new_12_vertices2'] = new_vertices3 
        return_dict[prefix+'pred_12_points2'] = pred_points3
        return_dict[prefix+'distortion3'] = distortions3 
        
        return_dict[prefix+'distortion'] = total_distortion 
        return new_points3, return_dict
    
    def compute_distortion(self, a, b, c):
        A = torch.zeros(a.shape[0], 3,3).cuda()
        
        A[:,0,0] = b[:,0,0]*a[:,0,0]
        A[:,0,1] = b[:,0,0]*a[:,0,1]
        A[:,0,2] = b[:,0,1]
        
        A[:,1,0] = c[:,0,0]*a[:,1,0] + c[:,0,1]*b[:,1,0]*a[:,0,0]
        A[:,1,1] = c[:,0,0]*a[:,1,1] + c[:,0,1]*b[:,1,0]*a[:,0,1]
        A[:,1,2] = c[:,0,1]*b[:,1,1]
        
        A[:,2,0] = c[:,1,0]*a[:,1,0] + c[:,1,1]*b[:,1,0]*a[:,0,0]
        A[:,2,1] = c[:,1,0]*a[:,1,1] + c[:,1,1]*b[:,1,0]*a[:,0,1]
        A[:,2,2] = c[:,1,1]*b[:,1,1]
        return A 

class TutteVaryingNormal(nn.Module):
    def __init__(self, mesh, lap_dict, normal_matrices, radius=1, prefix='',):
        super(TutteVaryingNormal, self).__init__()
        
        self.normal_matrices = normal_matrices.unsqueeze(1)
        self.num_layer = self.normal_matrices.shape[0]
        
        self.tutte_layers = nn.ModuleList([TutteLayer(mesh, lap_dict, radius, ) \
                for i in range(self.num_layer)])
        self.prefix = prefix 
    
    def forward(self, input_points):
        prefix = self.prefix
        new_points = input_points
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layer):
            rot_points = torch.matmul(self.normal_matrices[i],new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            cur_points = rot_points[:,:,0:2].clone() 
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points)
            new_points = torch.cat((pred_points, rot_points[:,:,2].unsqueeze(2)), dim=2) 
            
            new_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            cur_distortion = torch.eye(3).unsqueeze(0).repeat(distortions.shape[0],1,1).cuda()
            cur_distortion[:, :2,:2] = distortions 
            cur_distortion = torch.matmul(self.normal_matrices[i].transpose(1,2), cur_distortion)
            cur_distortion = torch.matmul(cur_distortion, self.normal_matrices[i], ) 
            
            if total_distortion is None: 
                total_distortion = cur_distortion 
            else: 
                total_distortion = torch.matmul(cur_distortion, total_distortion)

            return_dict[prefix+'pred_points_%d'%(i)] = new_points 
            return_dict[prefix+'new_vertices_%d'%(i)] = new_vertices
            return_dict[prefix+'pred_2d_points_%d'%(i)] = pred_points
            return_dict[prefix+'distortion%d'%(i+1)] = distortions
             
        return_dict[prefix+'distortion'] = total_distortion
        return new_points, return_dict
      
class TutteModel(nn.Module):
    def __init__(self, mesh, bound_verts, num_layer=2, radius=1):
        super(TutteModel, self).__init__()
        self.num_layers = num_layer
        self.mesh = mesh
        self.bound_verts = bound_verts
        
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

        self.lap_dict = {'interior_verts': self.interior_verts, 'inter_vert_mapping':self.inter_vert_mapping, 'bound_verts':self.bound_verts, \
            'bound_ids': self.bound_ids, 'bound_b_val_ids': self.bound_b_val_ids, 'interior_ids': self.interior_ids, 'lap_index': self.lap_index}
        
        # self.normals = np.array([[1,0,0], [0,1,0], [0,0,1]])
        # self.normals = np.array([[0.5,0.5,0], [0,0.5,0.5], [0.5,0,0.5]])
        
        self.normals = np.array([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                                 [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]], np.float32)
        
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        print("normals", self.normals)

        self.normal_matrices = np.zeros((self.normals.shape[0], 3,3))
        for i in range(self.normals.shape[0]):
            self.normal_matrices[i] = self.rotation_matrix_from_vectors(self.normals[i])

        self.normal_matrices = torch.from_numpy(self.normal_matrices).float().cuda()
        
  
        self.tutte_layers = nn.ModuleList([TutteVaryingNormal(mesh, self.lap_dict,self.normal_matrices, radius,  prefix='L%d_'%(i) ) \
                for i in range(num_layer)])
        
        # self.tutte_layers = nn.ModuleList([TutteTriplane(mesh, self.lap_dict, radius,  prefix='L%d_'%(i) ) \
        #         for i in range(num_layer)])
        
    def forward(self, input_points):
        pred_points = input_points.clone() 
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layers):
            pred_points, mid_dict = self.tutte_layers[i](pred_points)
            distortions = mid_dict['L%d_distortion'%(i)]
            if total_distortion is None:
                total_distortion = distortions 
            else: 
                total_distortion = torch.matmul(distortions, total_distortion)
            return_dict.update(mid_dict)
        return_dict['pred_points'] = pred_points
        return_dict['total_distortion'] = total_distortion
        # return_dict['new_vertices'] = new_vertices 
        return return_dict
    
    def rotation_matrix_from_vectors(self, vec1, vec2=[0,0,1]):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if any(v): #if not all zeros then 
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        else:
            return np.eye(3) #cross of all zeros only occurs on identical directions

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__=="__main__":
    
    name_id = 12 # 12, 14, 16
    fps_ratio = 0.01
    N = 25
    num_layer = 3
    
    d_weight = 0.00
    r_weight = 0.00
    
    temp_mesh = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/temp_pose.obj')
    mesh1 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/temp_leg.obj')
    # mesh1 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/temp_pose.obj')
    
    temp_mesh = Mesh(filename = '/home/bos/projects/amass/out_mesh/ACCAD/Female1Gestures_c3d/Female1 Subj Calibration_poses.obj')
    # mesh1 = Mesh(filename = '/home/bos/projects/amass/out_mesh/ACCAD/Female1Gestures_c3d/Female1 Subj Calibration_poses.obj')
    
    scale_factor = 1.4 # 1.6 
    vert0 = temp_mesh.v 
    
    vert1 = mesh1.v 
    # mesh2 = Mesh(filename = '/home/bos/projects/amass/out_mesh/ACCAD/Female1Running_c3d/C3 - Run_poses.obj')
    # mesh2 = Mesh(filename = '/home/bos/projects/amass/out_mesh/ACCAD/Female1General_c3d/A10 - lie to crouch_poses.obj')
    mesh2 = Mesh(filename = '/home/bos/projects/amass/out_mesh/ACCAD/Female1General_c3d/A8 - crouch to lie_poses.obj')
    # mesh2 = Mesh(filename = '/home/bos/projects/amass/out_mesh/BMLmovi/Subject_1_F_MoSh/Subject_1_F_18_poses.obj')

    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/down2_arm.obj')

    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/halfup_arm.obj')

    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/rand_pose_-0.5_0.5/%d.obj'%(name_id))

    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/rand_pose_-1_1/29.obj')
    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/rand_pose_1_leg/%d.obj'%(name_id))
    # mesh2 = Mesh(filename = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/rand_pose_1_arm/%d.obj'%(name_id))

    vert2 = mesh2.v 
    alpha = np.pi/2 
    rot_matrix = np.array([ [1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
    vert0 = np.matmul(vert0,rot_matrix )
    vert1 = np.matmul(vert1,rot_matrix )
    vert2 = np.matmul(vert2,rot_matrix )
    print(vert1.min(0),vert1.max(0))
    print(vert2.min(0),vert2.max(0))

    min1 = vert0.min(0)
    vert0 = vert0 - min1
    max1 = vert0.max(0)
    min1_1 = vert0.min(0)
    vert0 = vert0 / (max1[2] - min1_1[2])
    min1_2 = vert0.min(0) 
    max1_2 = vert0.max(0) 
    vert0 = (vert0 - (max1_2 -min1_2)/2)*scale_factor
    print('vert0',vert0.min(0),vert0.max(0))
    
    vert1 = vert1 - min1
    vert1 = vert1 / (max1[2] - min1_1[2])
    vert1 = (vert1 - (max1_2 -min1_2)/2)*scale_factor
    print('vert1',vert1.min(0),vert1.max(0))

    vert2 = vert2 - min1
    vert2 = vert2 / (max1[2] - min1_1[2])
    vert2 = (vert2 - (max1_2 -min1_2)/2)*scale_factor
    print('vert2', vert2.min(0),vert2.max(0))
    
    # out_path = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/big_arms/%d'%(name_id)
    # out_path = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/new_randpose_0.5/%d'%(name_id)
    # out_path = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/big_arms/down2_arm'
    # out_path = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/amass/BMLmovi_1_18'
    out_path = '/home/bos/projects/SMPL_python_v.1.1.0/smpl/smpl_webuser/hello_world/amass/a8'


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # input_points = torch.from_numpy(handle_points[::1,:]).float().unsqueeze(0)
    # target_points = torch.from_numpy(handle_target[::1,:]).float().unsqueeze(0)
    shape_points = torch.from_numpy(vert1).float().unsqueeze(0).cuda()
    
    # batch = torch.zeros(shape_points.shape[1]).long().cuda()
    # index = fps(shape_points[0], batch, ratio=fps_ratio, random_start=False).cpu().numpy()
    
    # index = [5664, 2235] # arm 2 
    # index = [2209, 5664, 250]
    # index = [248, 6469, 5109, 5569, 5880, 3008, 1574, 2425, 2922, 4532, 4532, 6740]
    
    # index = [5767, 5163, 4982, 3068, 444, 1864, 1695, 2306, 3065, 3077, 6293, 1324, 6382, 4418, 3475, 4362, 1366,4529, 
    #         1043, 6723, 3325, 6615, 3232, 38, 3939, 10, 4874, 674, 4561, 1075, 4361, 874, 5384, 2073]
    # handle_points = vert1[index]
    # handle_target = vert2[index]
    
    body_ids = [5767, 5163, 4982, 3068, 444, 1864, 1695, 2306, 3065, 3077, 6293, 1324, 6382, 4418, 3475, 4362, 1366,4529, 
            1043, 6723, 3325, 6615, 3232,   
            38, 3939, 10, 4874, 674, 4561, 1075, 4361, 874, 5384, 2073]
    back_ids = [274, 446, 487, 3978, 422, 453, 1823, 5274, 2947, 1909, 1940, 2469, 6405, 5371, 5526, 6008, 3007, 1269, 4211, 
           6472, 1786, 5250, 3159, 3088, 6512, 1460, 4930, 1049, 4535, 1466, 4939, 3435, 6730]
    side_ids = [250, 5083, 5292, 6282, 5673, 1861, 2821, 1912, 2002, 4334, 4571, 972, 1095, ]
    # handle_points = vert1[body_ids+back_ids+side_ids]
    # handle_target = vert2[body_ids+back_ids+side_ids]
    # handle_points = vert1[body_ids]
    # handle_target = vert2[body_ids]
    handle_points = vert1
    handle_target = vert2

    # prefix = 'h80_dr0.001_'
    prefix = 'normal_leg_all_'
    input_points = torch.from_numpy(handle_points).float().unsqueeze(0).cuda()
    target_points = torch.from_numpy(handle_target).float().unsqueeze(0).cuda()
    print(input_points.shape, target_points.shape, shape_points.shape)

    radius = 1
    vertices, faces, bound_verts = build_uniform_square_graph(N)
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

    tutte_model = TutteModel(mesh_input, bound_verts, num_layer=num_layer) 
    
    lr = 0.02 # 0.02 
    # optim =  torch.optim.SGD(tutte_model.parameters(), lr=lr, momentum=0.9)
    optim =  torch.optim.Adam(tutte_model.parameters(), lr=lr, weight_decay=0.0)

    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=10000)
    creteria = torch.nn.MSELoss()
    
    def model_func(x):
        return_dict = tutte_model(x)
        pred_points = return_dict['pred_points'] 
        return pred_points
    
    t1 = time.time() 
    for step in range(15000):
        
        input_points.requires_grad=True 
        optim.zero_grad()

        return_dict = tutte_model(input_points)
        if d_weight==0:
            return_dict1 = return_dict
        else:
            return_dict1 = tutte_model(shape_points)

        pred_points = return_dict['pred_points']   
        loss1 = creteria(pred_points, target_points)
        loss3 = 0.0 
        for i in range(num_layer):
            # loss3 += distortion_loss(return_dict1['L%d_distortion'%(i)])
            loss3 += distortion_loss(return_dict1['L%d_distortion1'%(i)],dim=2)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion2'%(i)],dim=2)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion3'%(i)],dim=2)/3

        loss3 = loss3 / num_layer 
        loss2 = distortion_loss(return_dict1['total_distortion'],dim=3)
        loss = loss1 + d_weight*loss2 + r_weight*loss3

        # jacob = torch.autograd.functional.jacobian(model_func, input_points)
        # import pdb; pdb.set_trace() 

        loss.backward()
        optim.step()
        scheduler.step()
        # print("forward: %.3f, loss: %.3f, backward: %.3f, optim: %.3f"%((t22-t11)*1000, (t33-t22)*1000, (t44-t33)*1000, (t55-t44)*1000),)

        if step%20==0:
            print(step,  'loss', loss.item(),loss1.item(), loss2.item(), loss3.item())

        if step%200==0:
            
            print(out_path, prefix)
            pred_points_np = return_dict1['pred_points'].detach().cpu().numpy()[0]
            alpha = -np.pi/2 
            rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
            pred_points_np = np.matmul(pred_points_np,rot_matrix )
            mesh3 = Mesh(v=pred_points_np, f=mesh1.f)
            mesh3.write_ply(os.path.join(out_path, prefix+'_%d_pred.ply'%(step)))
            mesh3 = Mesh(v=np.matmul(vert1,rot_matrix ), f=mesh1.f)
            mesh3.write_ply(os.path.join(out_path, prefix+'_input.ply'))
            mesh3 = Mesh(v=np.matmul(vert2,rot_matrix ), f=mesh1.f)
            mesh3.write_ply(os.path.join(out_path, prefix+'_target.ply'))
            
            # mesh3 = Mesh(v=np.matmul(input_points.detach().cpu().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle.ply'))
            # mesh3 = Mesh(v=np.matmul(pred_points.detach().cpu().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle_pred.ply'))
            # mesh3 = Mesh(v=np.matmul(target_points.detach().cpu().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle_gt.ply'))

            for ii in range(0):
                pred_points_np = return_dict1['L%d_pred_points3'%(ii)].detach().cpu().numpy()[0]
                alpha = -np.pi/2 
                rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
                pred_points_np = np.matmul(pred_points_np,rot_matrix )
                mesh3 = Mesh(v=pred_points_np, f=mesh1.f)
                mesh3.write_ply(os.path.join(out_path, prefix+'50_pred6_%d.ply')%(ii)) 

    t2 = time.time() 
    print('total training time', t2-t1)