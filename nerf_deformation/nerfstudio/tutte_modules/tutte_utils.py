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

import torch
import torch.nn as nn 
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import torch_geometric 
from torch_scatter import scatter
from torch_sparse_solve import solve

import igl
from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
import torch_sparse 


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


def get_initial_tutte_mesh(N):
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
    mesh_input['mesh_resolution'] = N
    return mesh_input, bound_verts



def load_density_shape(shape_path):
    shape_mesh = trimesh.load(shape_path)
    shape_points = shape_mesh.vertices
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


# scale_mesh = Mesh(filename='/home/bosun/projects/PCPerception/tools/template/smpl_template.obj')
scale_mesh = trimesh.load('/home/bosun/projects/PCPerception/tools/template/smpl_template.obj')
vert1 = torch.from_numpy(scale_mesh.vertices)
min1 = vert1.min(0)[0].unsqueeze(0)
vert1 = vert1 - min1

max1_1 = vert1.max(0)[0]
min1_1 = vert1.min(0)[0]
vert1 = vert1 / (max1_1[1] - min1_1[1])

min1_2 = vert1.min(0)[0].unsqueeze(0)
max1_2 = vert1.max(0)[0].unsqueeze(0)
vert1 = (vert1 - (max1_2 -min1_2)/2)*1.4

def density_to_learning(points, deform_shape):
    if deform_shape=='siming3_nerfacto':
        alpha = torch.tensor(-torch.pi/2)
        rot_matrix1 = torch.tensor([[1,0,0], [0,torch.cos(alpha), torch.sin(alpha)], [0, -torch.sin(alpha), torch.cos(alpha)],])
        alpha1 =  torch.tensor(-torch.pi/2) # y
        rot_matrix2 = torch.tensor([[torch.cos(alpha1),0, torch.sin(alpha1)], [0,1,0], [-torch.sin(alpha1),0, torch.cos(alpha1)]])
        alpha2 =  torch.tensor(-torch.pi/4) 
        rot_matrix3 = torch.tensor([[torch.cos(alpha2), torch.sin(alpha2), 0], [-torch.sin(alpha2), torch.cos(alpha2), 0], [0,0,1]])

        points = torch.matmul(points, rot_matrix3)
        points = torch.matmul(points, rot_matrix1)
        points = torch.matmul(points, rot_matrix2)
        points = points * 3.5
        points[:, 0] -= 0.05 # siming3
        # points[:, 1] -= 0.3 # siming3
        points[:, 2] += 0.22 # siming3
        vert_density = points 
        vert_density = vert_density - min1
        vert_density = vert_density / (max1_1[1] - min1_1[1])
        vert_density = (vert_density - (max1_2 - min1_2)/2) * 1.4
    else:
        alpha = torch.tensor(-torch.pi/2)
        rot_matrix1 = torch.tensor([[1,0,0], [0,torch.cos(alpha), torch.sin(alpha)], [0, -torch.sin(alpha), torch.cos(alpha)]]).float()

        vert_density = points
        vert_density = torch.matmul(vert_density, rot_matrix1)
        if deform_shape=='girl2':
            vert_density[:, 1] -= 0.3
            vert_density = vert_density * 1.5 # girl
        elif deform_shape=='robo':
            # print('##robo')
            vert_density[:, 1] -= 0.25 # robo
            vert_density[:, 0] -= 0.06 # robo
        else:
            print('>??????')
        vert_density = vert_density - min1
        vert_density = vert_density / (max1_1[1] - min1_1[1])
        vert_density = (vert_density - (max1_2 - min1_2)/2) * 1.4
    return vert_density

def learning_to_density(points, deform_shape):
    if deform_shape=='siming3_nerfacto':
        alpha = torch.tensor(-torch.pi/2)
        rot_matrix1 = torch.tensor([[1,0,0], [0,torch.cos(alpha), torch.sin(alpha)], [0, -torch.sin(alpha), torch.cos(alpha)],])
        alpha1 =  torch.tensor(-torch.pi/2) # y
        rot_matrix2 = torch.tensor([[torch.cos(alpha1),0, torch.sin(alpha1)], [0,1,0], [-torch.sin(alpha1),0, torch.cos(alpha1)]])
        alpha2 =  torch.tensor(-torch.pi/4) 
        rot_matrix3 = torch.tensor([[torch.cos(alpha2), torch.sin(alpha2), 0], [-torch.sin(alpha2), torch.cos(alpha2), 0], [0,0,1]])

        vert = points
        vert = vert / 1.4
        vert = vert + (max1_2 - min1_2)/2
        vert = vert * (max1_1[1] - min1_1[1])
        vert += min1

        vert[:,2] -= 0.22 
        vert[:,0] += 0.05 
        vert = vert / 3.5 
        vert = torch.matmul(vert.float(), rot_matrix2.transpose(0,1))
        vert = torch.matmul(vert.float(), rot_matrix1.transpose(0,1))
        vert = torch.matmul(vert.float(), rot_matrix3.transpose(0,1))
    else:
        vert = points
        vert = vert /1.4
        vert = vert + (max1_2 - min1_2)/2
        vert = vert * (max1_1[1] - min1_1[1])
        vert += min1

        alpha = torch.tensor(-torch.pi/2)
        rot_matrix1 = torch.tensor([[1,0,0], [0,torch.cos(alpha), torch.sin(alpha)], [0, -torch.sin(alpha), torch.cos(alpha)]]).float()
        if deform_shape=='girl2':
            vert = vert / 1.5
            vert[:, 1] += 0.3
        elif deform_shape=='robo':
            vert[:, 1] += 0.25 # robo
            vert[:, 0] += 0.06 # robo

        vert = torch.matmul(vert.float(), rot_matrix1.transpose(0,1))
    return vert
    
