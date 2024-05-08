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
# from psbody.mesh import Mesh

import torch
import torch.nn as nn 
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import torch_geometric 
from torch_scatter import scatter
from torch_sparse_solve import solve
# from cholespy import CholeskySolverF, MatrixType

import igl
from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
import torch_sparse 


def load_density_shape(shape_path):
    shape_mesh = trimesh.load(shape_path)
    #shape_mesh = filter_outliers('./keep_shapes/trex_shape.ply')
    #shape_points = shape_mesh.v

    # shape_mesh = trimesh.load('./keep_shapes/girl_shape_5.ply')
    # shape_mesh = trimesh.load('./keep_shapes/ficus_shape.ply')
    # shape_mesh1 = trimesh.load('./keep_shapes/ficus_shape_surf.ply')
    # shape_points = np.concatenate((shape_mesh.vertices, shape_mesh1.vertices),axis=0)
    # shape_mesh = trimesh.load('./keep_shapes/mic_dense_mesh_10.ply')

    # shape_mesh = trimesh.load('/home/bos/projects/nerf/exports/siming3/mesh/dense_mesh_10.ply')
    # shape_mesh = trimesh.load('/home/bosun/projects/nerf/positions/siming3_nerfacto/dense_mesh_10_small.ply')
    # shape_mesh = trimesh.load('/home/bosun/projects/nerf/keep_shapes/dense_mesh_10_small.ply')
    # shape_mesh = trimesh.load('/home/bosun/projects/nerf/exports/veh1/mesh/dense_mesh_1_simp.ply')
    # shape_mesh = trimesh.load('/home/bosun/projects/nerf/exports/giraff/mesh/dense_mesh_60_simp_nofloor.ply')
    shape_points = shape_mesh.vertices
    # shape_points = np.concatenate((shape_mesh.vertices, shape_mesh1.vertices),axis=0)
    # shape_points = trimesh.load('./keep_shapes/trex_dense.ply').vertices
    shape_points = shape_points/2 + 0.5
    shape_points_cuda = torch.from_numpy(shape_points).float().cuda()
    batch_y = torch.zeros(shape_points_cuda.shape[0]).cuda()
    print('loaded keep shape points from', shape_path)
    return shape_points_cuda, batch_y 

def get_points_scaling_data(shape_points):
    if shape_points is None:
        return None 
    shape_points = (shape_points.cpu() - 0.5)*2 
    min0 = shape_points.min(0)[0].unsqueeze(0)
    max0 = shape_points.max(0)[0].unsqueeze(0)
    scale = min(min(1.4/(max0-min0)))
    return (min0, max0, scale)


def density_to_tutte(points, points_scaling_data, shape_name):
    if shape_name not in ['giraff', 'veh1']:
        return points 
    if points_scaling_data is None:
        return points 
    min0, max0, scale = points_scaling_data 
    points = points - min0- (max0-min0)/2
    points = points * scale 
    return points

def tutte_to_density(points, points_scaling_data, shape_name):
    if shape_name not in ['giraff', 'veh1']:
        return points 
    if points_scaling_data is None:
        return points 
    min0, max0, scale = points_scaling_data 

    points = points / scale 
    points = points + min0 + (max0-min0)/2
    return points



# def density_to_tutte(points, shape_name):
#     if shape_name=='veh1':
#         max0 = torch.tensor([[0.007933, 0.12973499, -0.26117599]])
#         min0 = torch.tensor([[-0.12119, -0.086037, -0.356814]])
#         scale = 6.488330410321643 
#     elif shape_name=='giraff':
#         print('giraff')
#         max0 = torch.tensor([[ 0.210859,  0.079308, -0.059991]])
#         min0 = torch.tensor([[ 0.003601,  -0.185434,  -0.33868501]])
#         scale = 5.023430580115293
#     else:
#         return points 
#     points = points - min0- (max0-min0)/2
#     points = points * scale 
#     return points

# def tutte_to_density(points, shape_name):
#     if shape_name=='veh1':
#         max0 = torch.tensor([[0.007933, 0.12973499, -0.26117599]])
#         min0 = torch.tensor([[-0.12119, -0.086037, -0.356814]])
#         scale = 6.488330410321643 
#     elif shape_name=='giraff':
#         print('giraff')
#         max0 = torch.tensor([[ 0.210859,  0.079308, -0.059991]])
#         min0 = torch.tensor([[ 0.003601,  -0.185434,  -0.33868501]])
#         scale = 5.023430580115293
#     else:
#         return points 
#     points = points / scale 
#     points = points + min0 + (max0-min0)/2
#     return points


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

class TutteLayer(nn.Module):
    def __init__(self, mesh, lap_dict, radius=1, use_sigmoid=True, circle_map=False):
        super(TutteLayer, self).__init__()
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.radius = radius
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
        W_var = torch.empty(self.batch_size, n_edges)
        torch.nn.init.uniform_(W_var)
        self.W_var = nn.Parameter(W_var)
        
        angle_var = torch.empty(self.batch_size, len(self.bound_verts))
        torch.nn.init.uniform_(angle_var)
        self.angle_var = nn.Parameter(angle_var) 

    def forward(self, input_points, inverse=False):
        if inverse:
            return self.forward_inverse(input_points)
        # W_var = var_pred[:, :len(self.edges)]
        # angle_var = var_pred[:, len(self.edges):]
        
        input_points_np = input_points.detach().numpy() 
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

    def forward_inverse(self, input_points,):
        
        new_vertices = self.tutte_embedding_sparse() 
        
        input_points_np = input_points.detach().numpy() 
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        
        face_ids = np.zeros((batch_size, N_points), np.int32)
        
        for i in range(batch_size):
            tri = Delaunay(new_vertices[i].detach().numpy())
            tri.simplices = np.array(self.faces, np.int32)
            input_points_np_i = input_points_np[i] 
            face_ids_i = Delaunay.find_simplex(tri, input_points_np_i.astype(np.float64), bruteforce=True) 
            face_ids[i] = face_ids_i 
            
        input_nodes = self.faces[face_ids.reshape(-1)] 
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)
        
        input_areas = self.get_areas_inverse(input_points, input_nodes, new_vertices) # [b, N, 3]
        pred_points = self.get_tutte_from_triangle(self.vertices.unsqueeze(0).repeat(batch_size, 1, 1), input_nodes, input_areas)
        
        distortions = torch.zeros(batch_size* N_points, 2,2)
        
        return pred_points, new_vertices, distortions
    
    def tutte_embedding_sparse(self, ):
        """
        Args:
            W_var: [b, n_edge]
            angle_var: [b, n_bound]
        """
        n_vert = self.vertices.shape[0]
        W_var = torch.sigmoid(torch.clamp(self.W_var, -10, 10))*(1-0.4) + 0.2
        
        # if self.use_sigmoid:
        #     W_var = torch.sigmoid(torch.clamp(self.W_var, -10, 10))
        # else:
        #     W_var = torch.clamp(self.W_var, 0.05, 1) 
        batch_size = W_var.shape[0]
        angle_var = torch.sigmoid(torch.clamp(self.angle_var, -10, 10))*(1-0.4) + 0.2
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
        bound_pos = torch.zeros((batch_size * angle_var.shape[1], 2)) # [b, n_bound, 2]

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
        
        lap_values = torch.zeros((batch_size, self.lap_size, ))
        
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
        
        batch_dim = torch.zeros((batch_size * lap_index.shape[1])).long()  
        for bi in range(batch_size):
            batch_dim[bi*lap_index.shape[1]: (bi+1)*lap_index.shape[1]] = bi 
        
        # lap_index = torch.cat((torch.zeros(1, lap_index.shape[1]), lap_index.cpu()), dim=0).long() 
        lap_indices = torch.cat((batch_dim.unsqueeze(0), lap_index.repeat(1, batch_size)), dim=0).long()  
        
        A = torch.sparse_coo_tensor(lap_indices, lap_values.double(), (batch_size, len(self.interior_verts), len(self.interior_verts)))
        b = b[:, self.interior_verts].double()        
        x =  solve(A, b) 
        
        out_pos = torch.zeros(batch_size, n_vert, 2) 
        out_pos[:, self.bound_verts] = bound_pos.float()
        out_pos[:, self.interior_verts] = x.float()
        
        return out_pos 
    
    def get_areas(self, points, tri_nodes):
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
        areas = torch.zeros(tri_nodes.shape)
        areas[:,:, 0] = area_A 
        areas[:,:, 1] = area_B
        areas[:,:, 2] = area_C 
        
        return areas
    
    def get_areas_inverse(self, points, tri_nodes, vert):
        """

        Args:
            points: [b, N, 3]
            tri_nodes: [b, N, 3]
        """
        areas = torch.zeros(tri_nodes.shape)
        batch_size, n_vert, _ = vert.shape 
        n_points = points.shape[1]
        
        vertices = vert.view(batch_size * n_vert, 2)
        offset = 0 
        for i in range(batch_size):
            tri_nodes[i] += n_vert * i  
            
        tri_nodes = np.reshape(tri_nodes, (batch_size * n_points, 3))
    
        loc = points
        init_A = vertices[tri_nodes[:,0].astype(np.int32), :].view(batch_size, n_points, 2)
        init_B = vertices[tri_nodes[:,1].astype(np.int32), :].view(batch_size, n_points, 2)
        init_C = vertices[tri_nodes[:,2].astype(np.int32), :].view(batch_size, n_points, 2)
    
        area_A = torch.abs((loc-init_B)[:,:,0]*(loc-init_C)[:,:,1] - (loc-init_B)[:,:,1]*(loc-init_C)[:,:,0])/2
        area_B = torch.abs((loc-init_A)[:,:,0]*(loc-init_C)[:,:,1] - (loc-init_A)[:,:,1]*(loc-init_C)[:,:,0])/2
        area_C = torch.abs((loc-init_A)[:,:,0]*(loc-init_B)[:,:,1] - (loc-init_A)[:,:,1]*(loc-init_B)[:,:,0])/2
        
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
        A = torch.zeros(batch_size*n_points, 6, 6)
        b = torch.zeros(batch_size*n_points, 6)
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
    
    def forward(self, input_points, inverse=False, hide=False):
        if inverse:
            return self.forward_inverse(input_points, hide=hide)
        
        prefix = self.prefix
        new_points = input_points
    
        cur_points1 = new_points[:,:,0:2].clone() 
        # pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1, inverse=inverse)
        pred_points1 = cur_points1
        new_points1 = torch.cat((pred_points1, new_points[:,:,2].unsqueeze(2)), dim=2)
        
        cur_points2 = torch.cat((new_points1[:,:,0].unsqueeze(2), new_points1[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2, inverse=inverse)
        # pred_points2 = cur_points2
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points1[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2) 
        
        cur_points3 = new_points2[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3, inverse=inverse)
        new_points3 = torch.cat((new_points2[:,:,0].unsqueeze(2), pred_points3), dim=2)
        
        # total_distortion = self.compute_distortion(distortions1, distortions2,distortions3)
        
        return_dict = {} 
        return_dict[prefix+'pred_points1'] = new_points1 
        # return_dict[prefix+'new_01_vertices1'] = new_vertices1
        # return_dict[prefix+'pred_01_points1'] = pred_points1
        # return_dict[prefix+'distortion1'] = distortions1 
        
        return_dict[prefix+'pred_points2'] = new_points2
        # return_dict[prefix+'new_02_vertices2'] = new_vertices2 
        # return_dict[prefix+'pred_02_points2'] = pred_points2
        # return_dict[prefix+'distortion2'] = distortions2
        
        return_dict[prefix+'pred_points3'] = new_points3
        # return_dict[prefix+'new_12_vertices2'] = new_vertices3 
        # return_dict[prefix+'pred_12_points2'] = pred_points3
        # return_dict[prefix+'distortion3'] = distortions3 
        
        # return_dict[prefix+'distortion'] = total_distortion 
        return new_points3, return_dict
    
    def forward_inverse(self, input_points, hide=False):
        prefix = self.prefix
        new_points = input_points
        
        cur_points3 = new_points[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3, inverse=True)
        new_points3 = torch.cat((new_points[:,:,0].unsqueeze(2), pred_points3), dim=2)
        cur_points2 = torch.cat((new_points3[:,:,0].unsqueeze(2), new_points3[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2, inverse=True)
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points3[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2) 
        
        cur_points1 = new_points2[:,:,0:2].clone() 
        pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1, inverse=True)
        new_points1 = torch.cat((pred_points1, new_points2[:,:,2].unsqueeze(2)), dim=2)
        
        return_dict = {} 
        return_dict[prefix+'pred_points1'] = new_points1 
        # return_dict[prefix+'new_01_vertices1'] = new_vertices1
        # return_dict[prefix+'pred_01_points1'] = pred_points1
        # return_dict[prefix+'distortion1'] = distortions1 
        
        return_dict[prefix+'pred_points2'] = new_points2
        # return_dict[prefix+'new_02_vertices2'] = new_vertices2 
        # return_dict[prefix+'pred_02_points2'] = pred_points2
        # return_dict[prefix+'distortion2'] = distortions2
        
        return_dict[prefix+'pred_points3'] = new_points3
        # return_dict[prefix+'new_12_vertices2'] = new_vertices3 
        # return_dict[prefix+'pred_12_points2'] = pred_points3
        # return_dict[prefix+'distortion3'] = distortions3 
        
        # return_dict[prefix+'distortion'] = total_distortion    
        return new_points1, return_dict
    
        
    def compute_distortion(self, a, b, c):
        A = torch.zeros(a.shape[0], 3,3)
        
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
    
    def forward(self, input_points, inverse=False):
        if inverse:
            return self.forward_inverse(input_points)
        
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
            
            cur_distortion = torch.eye(3).unsqueeze(0).repeat(distortions.shape[0],1,1)
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
    
    def forward_inverse(self, input_points):
        prefix = self.prefix
        new_points = input_points
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layer-1, -1, -1):
            # print(i)
            rot_points = torch.matmul(self.normal_matrices[i], new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            cur_points = rot_points[:,:,0:2].clone() 
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points, inverse=True)
            new_points = torch.cat((pred_points, rot_points[:,:,2].unsqueeze(2)), dim=2) 
            new_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
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

        self.inter_vert_mapping = torch.zeros(len(self.vertices)).long()
        self.inter_vert_mapping[self.interior_verts] = torch.arange(len(self.interior_verts))

        n_edges = self.edges.shape[1]
        W_var = torch.ones(n_edges)
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

        self.bound_b_val_ids = -1*torch.ones(len(self.vertices), len(self.bound_verts)).long()
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

        self.normal_matrices = torch.from_numpy(self.normal_matrices).float()
        
        self.tutte_layers = nn.ModuleList([TutteTriplane(mesh, self.lap_dict, radius,  prefix='L%d_'%(i) ) \
                for i in range(num_layer)])
        # self.tutte_layers = nn.ModuleList([TutteVaryingNormal(mesh, self.lap_dict,self.normal_matrices, radius,  prefix='L%d_'%(i) ) \
        #         for i in range(num_layer)])
        

    def forward(self, input_points, inverse=False):
        if inverse:
            return self.forward_inverse(input_points)
        
        pred_points = input_points.clone() 
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layers):
            pred_points, mid_dict = self.tutte_layers[i](pred_points, inverse=inverse) 
            
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
    
    def forward_inverse(self, input_points):
        pred_points = input_points.clone() 
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layers):
        
            pred_points, mid_dict = self.tutte_layers[self.num_layers-i-1](pred_points, inverse=True) 
            return_dict.update(mid_dict)
        return_dict['pred_points'] = pred_points
        return_dict['total_distortion'] = None 
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

def distortion_loss(distortion, dim=3):
    # [N, 2,2]
    d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
    loss = torch.square(d - torch.eye(dim).unsqueeze(0).repeat(d.shape[0], 1,1)).mean()  
    return loss  

def jacobian_loss(pred_points, target_J):
    pred_J = _multiply_sparse_2d_by_dense_3d(igl_grad, pred_points).type_as(pred_points)
    pred_J = pred_J.view(pred_points.shape[0], -1, 3,3).transpose(2,3)
    loss = torch.square(pred_J - target_J).mean() 
    return loss 

if __name__=="__main__":

    temp_mesh = Mesh(filename = '/home/bosun/projects/nerf/tmp_tsdf/mesh_0.ply')
    mesh1 = temp_mesh 
    vert1 = temp_mesh.v 

    vert1 = vert1 * 0.7

    print(vert1.max(0), vert1.min(0))
    print(vert1.shape)

    handle_ids1 = [17275, 15162, 12685, 9697, 5908, 4714]
    handle_ids2 = [13923, 13533, 8328, 9299, 10911, 4254, 4149]


    handle_points = vert1[handle_ids1+handle_ids2]
    handle_target = vert1[handle_ids1+handle_ids2]
    handle_target[:6, 2] -= 0.4 
    vert2 = vert1 
    # handle_target = vert2

    out_path = '/home/bosun/projects/3D_Bijective_Mapping/lego'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    shape_points = torch.from_numpy(vert1).float().unsqueeze(0)

    input_points = torch.from_numpy(handle_points[::1,:]).float().unsqueeze(0)
    target_points = torch.from_numpy(handle_target[::1,:]).float().unsqueeze(0)

    print(input_points.shape, target_points.shape, shape_points.shape)

    N =  11
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
    mesh_input['vertices'] = torch.from_numpy(vertices).float()
    mesh_input['edges'] = torch.from_numpy(edges).long() 
    mesh_input['faces'] = faces 

    num_layer = 8
    tutte_model = TutteModel(mesh_input, bound_verts, num_layer=num_layer) 

    lr = 0.02 # 0.02 
    optim =  torch.optim.Adam(tutte_model.parameters(), lr=lr, weight_decay=0.0)

    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=5000)
    creteria = torch.nn.MSELoss()

    def model_func(x):
        return_dict = tutte_model(x)
        pred_points = return_dict['pred_points'] 
        return pred_points

    d_weight = 0.001
    r_weight = 0.001
    j_weight = 0.0 

    prefix = 'triplane_'
    t1 = time.time() 
    start_step = 0 

    model_path = 'lego_model.pt'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        tutte_model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
        print('load model from step', start_step)

    for step in range(start_step,10000):
        # print(step)
        input_points.requires_grad=True 
        optim.zero_grad()

        return_dict = tutte_model(input_points)
        rand_ids = np.random.randint(0, shape_points.shape[1], size=8000)
        return_dict1 = tutte_model(shape_points[:,rand_ids, :])
        # print('gg')
        # return_dict1 = return_dict 

        pred_points = return_dict['pred_points']   
        loss1 = creteria(pred_points, target_points)
        # loss_J = jacobian_loss(pred_points, target_grad)
        loss_J = torch.tensor([0.0]) 

        loss3 = 0.0 
        for i in range(num_layer):
            # loss3 += distortion_loss(return_dict1['L%d_distortion'%(i)])
            loss3 += distortion_loss(return_dict1['L%d_distortion1'%(i)],dim=2)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion2'%(i)],dim=2)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion3'%(i)],dim=2)/3

        loss3 = loss3 / num_layer 
        loss2 = distortion_loss(return_dict1['total_distortion'],dim=3)
        loss = loss1 + d_weight*loss2 + r_weight*loss3 + j_weight * loss_J  

        loss.backward()
        optim.step()
        scheduler.step()
        # print("forward: %.3f, loss: %.3f, backward: %.3f, optim: %.3f"%((t22-t11)*1000, (t33-t22)*1000, (t44-t33)*1000, (t55-t44)*1000),)

        if step%20==0:
            print(step, 'loss', loss.item(),loss1.item(), loss_J.item(),  loss2.item(), loss3.item())

        if step%200==0:
            return_dict_inv = tutte_model(return_dict1['pred_points'], inverse=True)
            pred_points_inv = return_dict_inv['pred_points']   
            print("hh")

            # print(out_path)
            pred_points_np = return_dict1['pred_points'].detach().cpu().numpy()[0]
            # alpha = -np.pi/2 
            alpha = 0
            rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
            pred_points_np = np.matmul(pred_points_np,rot_matrix )

            # mesh3 = Mesh(v=pred_points_np, f=mesh1.f)
            mesh3 = Mesh(v=pred_points_np, f=[])
            mesh3.write_ply(os.path.join(out_path, prefix+'_%d_pred.ply'%(step)))
            mesh3 = Mesh(v=np.matmul(vert1,rot_matrix ), f=mesh1.f)
            mesh3.write_ply(os.path.join(out_path, prefix+'_input.ply'))
            mesh3 = Mesh(v=pred_points_inv.detach().cpu().numpy()[0], f=[])
            mesh3.write_ply(os.path.join(out_path, prefix+'_pred_inv.ply'))
            # mesh3 = Mesh(v=np.matmul(vert2,rot_matrix ), f=mesh1.f)
            # mesh3.write_ply(os.path.join(out_path, prefix+'_target.ply'))

            # mesh3 = Mesh(v=np.matmul(input_points.detach().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle.ply'))
            # mesh3 = Mesh(v=np.matmul(pred_points.detach().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle_pred.ply'))
            # mesh3 = Mesh(v=np.matmul(target_points.detach().numpy()[0],rot_matrix ), f=[])
            # mesh3.write_ply(os.path.join(out_path, prefix+'_handle_gt.ply'))

            for ii in range(0):
                pred_points_np = return_dict1['L%d_pred_points3'%(ii)].detach().cpu().numpy()[0]
                alpha = -np.pi/2 
                rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
                pred_points_np = np.matmul(pred_points_np,rot_matrix )
                mesh3 = Mesh(v=pred_points_np, f=mesh1.f)
                mesh3.write_ply(os.path.join(out_path, prefix+'_pred_%d.ply')%(ii)) 

            torch.save({
                'step': step,
                'model_state_dict': tutte_model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss,
                }, 'lego_model.pt')


    t2 = time.time() 
    print('total training time', t2-t1)
