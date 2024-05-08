import time
import os 
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
import torch_cluster

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import NearestNeighbors

import matplotlib 
 
from nerfstudio.tutte_modules.tutte_model import * 
from nerfstudio.tutte_modules.tutte_utils import * 
from nerfstudio.tutte_modules.point_utils import * 
from nerfstudio.tutte_modules.tutte_train_utils import * 

random.seed(10)

device='cuda:0'
N =  11
num_layer = 8
scale_factor2 = 1

num_sample_body = 10000 
num_sample_handle = 5000 
num_sample_free = 15000
num_density_sample = 10000

d_weight = 0.003
r_weight = 0.003
j_weight = 0.1 
d_weight_or = d_weight
r_weight_or = r_weight
weighted = True 
print_loss = False 

model_name = 'up_rot'
##### You have to set the handle rotations and translations here #######
target_rotation_angles = np.array([-0.65, 0., 0.])*np.pi
trans_vec = np.array([[0., 0.45, 0.1]]) # tmp3 

density_path = '/home/bosun/projects/nerf/dump_densities/lego/'
out_path = './nerf_deformation/lego/%s'%(model_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)

model_dir = './nerf_deformation_models/lego'
model_path = model_dir +'/%s'%(model_name)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# create the body and handle by hand 
point_mesh = filter_outliers(path=os.path.join(density_path,'dense_mesh.ply') , thres=0.02) 
body_mesh = filter_outliers(path=os.path.join(density_path,'dense_body.ply'), thres=0.02) 
handle_mesh = filter_outliers(path=os.path.join(density_path,'dense_handle.ply'), thres=0.02) 

dense_points = point_mesh.vertices 
rand_ids = np.random.randint(0, dense_points.shape[0], size=20000) 
vert1 = dense_points[rand_ids] * scale_factor2 
translated_handle, total_rotation_matrix = set_handle_target(handle_mesh,target_rotation_angles, trans_vec)
print('Finish loading density points')

if not os.path.exists(os.path.join(density_path,'dense_free.ply')):
    moving_points = get_free_points(point_mesh, body_mesh, handle_mesh)
    free_mesh = trimesh.Trimesh(vertices=moving_points, faces=[])
    free_mesh.export(os.path.join(density_path,'dense_free.ply'))
    print('Finish getting free points')
else:
    free_mesh = trimesh.load(os.path.join(density_path,'dense_free.ply'))
    moving_points = free_mesh.vertices 


# build log 
log_txt = open(os.path.join(out_path, 'log.txt'), 'a')
log_str = 'rot_x: %.3f rot_y: %.3f rot_y: %.3f trans_x: %.3f trans_y: %.3f trans_z: %.3f \n\n'%(\
           target_rotation_angles[0], target_rotation_angles[0], target_rotation_angles[0], trans_vec[0][0], trans_vec[0][1], trans_vec[0][2])
log_txt.write(log_str)
log_txt.flush()

# move points to pytorch
shape_points = torch.from_numpy(vert1).float().unsqueeze(0)
density_points = torch.from_numpy(point_mesh.vertices *scale_factor2 ).float().unsqueeze(0)
body_points = torch.from_numpy(body_mesh.vertices *scale_factor2).float().unsqueeze(0)
handle_points = torch.from_numpy(handle_mesh.vertices*scale_factor2).float().unsqueeze(0)
free_points = torch.from_numpy(moving_points *scale_factor2).float().unsqueeze(0)
handle_target = torch.from_numpy(translated_handle*scale_factor2).float().unsqueeze(0)
total_rotation_torch = torch.from_numpy(total_rotation_matrix).float()

density_points_cuda = torch.from_numpy(point_mesh.vertices).float().to(device)
shape_points_cuda = torch.from_numpy(vert1).float().to(device)
batch_y = torch.zeros(shape_points_cuda.shape[0]).to(device)
batch_x = torch.zeros(density_points_cuda.shape[0]).to(device)
print('Finish setting up input')

#  Tutte model and Training parameters 
mesh_input, bound_verts = get_initial_tutte_mesh(N)
tutte_model = TutteModel(mesh_input, bound_verts, num_layer=num_layer) 

lr = 0.01
optim =  torch.optim.Adam(tutte_model.parameters(), lr=lr, weight_decay=0.0)
scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=5000)
creteria = torch.nn.MSELoss()
print('Finish setting up model')

start_step = 0 
num_handle = handle_points.shape[1]
num_body = body_points.shape[1]
num_free = free_points.shape[1]
num_shape = shape_points.shape[1]
loss2_best = 100 
loss2_i = torch.ones(num_shape)

for step in range(start_step,5000):
    optim.zero_grad()
    # sample volume points with large distortion 
    if step<2 or step%20==0:
        return_dict_tmp = tutte_model(shape_points) 
        loss2_tmp, loss2_i_tmp = distortion_loss(return_dict_tmp['total_distortion'],dim=3, weighted=weighted, return_i=True)
        shape_mask = loss2_i_tmp > 0.1
        masked_shape_points_cuda = shape_points_cuda[shape_mask]
        batch_y = torch.zeros(masked_shape_points_cuda.shape[0]).cuda()
        nearest_ids = torch_cluster.radius(density_points_cuda, masked_shape_points_cuda, r=0.05, batch_x=batch_x, batch_y=batch_y, max_num_neighbors=3000).cpu()
        nearest_ids = torch.unique(nearest_ids[1])
        sampled_density_or = density_points[:, nearest_ids, :]
    
    num_dens = sampled_density_or.shape[1]
    density_sample_ids = np.random.randint(0, num_dens, size=min(num_density_sample, num_dens) )
    sampled_density = sampled_density_or[:, density_sample_ids, :]

    body_ids = torch.randint(0, num_body, size=(min(num_sample_body, num_body),))
    handle_ids = torch.randint(0, num_handle, size=(min(num_sample_handle, num_handle),))    
    free_ids = torch.randint(0, num_free, size=(min(num_sample_free, num_free),)) 
    
    forward_points = torch.cat((handle_points[:, handle_ids], body_points[:,body_ids],
                               free_points[:, free_ids], sampled_density), dim=1)
    
    return_dict = tutte_model(forward_points) 
    pred_points = return_dict['pred_points'][:, :len(handle_ids), :] 
    loss1 = l2_loss(pred_points, handle_target[:, handle_ids])
    
    pred_points_fix = return_dict['pred_points'][:, len(handle_ids):len(handle_ids)+len(body_ids), :] 
    loss1 += l2_loss(pred_points_fix, body_points[:,body_ids])
    
    ### Jacobian loss 
    loss_j = 0.0 
    loss_j += vol_jacob_loss(return_dict['total_distortion'][0:len(handle_ids)], total_rotation_torch.transpose(0,1)) * num_handle/num_body
    loss_j += vol_jacob_loss(return_dict['total_distortion'][len(handle_ids):len(handle_ids)+len(body_ids)], torch.eye(3))

    loss2, loss2_i = distortion_loss(return_dict['total_distortion'],dim=3, weighted=weighted, print_loss=print_loss, return_i=True)
    loss3 = 0.0 
    for i in range(num_layer):
        loss3 += distortion_loss(return_dict['L%d_distortion1'%(i)],dim=2,)/3
        loss3 += distortion_loss(return_dict['L%d_distortion2'%(i)],dim=2,)/3
        loss3 += distortion_loss(return_dict['L%d_distortion3'%(i)],dim=2,)/3
    loss3 = loss3 / num_layer 
        
    d_weight = max(d_weight_or - 0.001 * (step // 600), 0.001)
    r_weight = max(r_weight_or * (0.5)**(step // 600), 0.001)
    loss = loss1 + d_weight*loss2 + r_weight*loss3 + j_weight * loss_j 
    
    loss.backward()
    optim.step()
    scheduler.step()
    
    if step%100==0:
        alpha = 0
        rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
        
        pred_points_shape_np = return_dict_tmp['pred_points'].detach().cpu().numpy()[0]
        mesh3 = trimesh.Trimesh(vertices=pred_points_shape_np, faces=[])
        mesh3.export(os.path.join(out_path, '%d_pred_shape_%.4f.ply'%(step, 100*loss.item())))
        
        loss1_i = torch.square(pred_points[0] - handle_target[0, handle_ids]).mean(1)
        pred_points_handle_np = pred_points.detach().cpu().numpy()[0]
        write_ply_color(pred_points_handle_np, torch.clamp(loss1_i, 0, 0.0002).detach().numpy()/0.0002, os.path.join(out_path, '%d_pred_handle_l2_%.4f.ply'%(step, 1000*loss1.item())))
        
        loss1_i = torch.square(pred_points_fix[0] - body_points[0,body_ids]).mean(1)
        pred_points_body_np = pred_points_fix.detach().cpu().numpy()[0]
        write_ply_color(pred_points_body_np, torch.clamp(loss1_i, 0, 0.0002).detach().numpy()/0.0002, os.path.join(out_path, '%d_pred_body_l2_%.4f.ply'%(step, 1000*loss1.item())))

        current_id = len(handle_ids)+len(body_ids)
        pred_points_free_np = return_dict['pred_points'][:, current_id: current_id+len(free_ids), :].detach().cpu().numpy()[0]
        write_ply_color(pred_points_free_np, torch.clamp(loss2_i[current_id:current_id+len(free_ids)], 0, 0.04).detach().numpy()/0.04, os.path.join(out_path, '%d_pred_free_d%.4f.ply'%(step, loss2.item())))
        
        if step==0:
            mesh3 = trimesh.Trimesh(vertices=np.matmul(body_points.detach().numpy()[0],rot_matrix ), faces=[])
            mesh3.export(os.path.join(out_path, 'body.ply'))
            mesh3 = trimesh.Trimesh(vertices=np.matmul(handle_points.detach().numpy()[0],rot_matrix ), faces=[])
            mesh3.export(os.path.join(out_path, 'handle.ply'))
            mesh3 = trimesh.Trimesh(vertices=np.matmul(handle_target.detach().numpy()[0],rot_matrix ), faces=[])
            mesh3.export(os.path.join(out_path, 'handle_target.ply'))
            mesh3 = trimesh.Trimesh(vertices=np.matmul(free_points.detach().numpy()[0],rot_matrix ), faces=[])
            mesh3.export(os.path.join(out_path, 'free.ply'))


    if step%100==0 and step>200:
        torch.save({
            'step': step,
            'model_state_dict': tutte_model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss,
            }, model_path +'/%s_step%d.pt'%(model_name, step))
        
    if step%20==0:
        cur_lr = optim.param_groups[-1]['lr']
        log_str = 'step: %d, loss: %.8f, loss_l2: %.9f,loss_j: %.4f, loss_d: %.4f, loss_r: %.4f, lr: %.4f\n'%(\
            step, loss.item(),loss1.item(), loss_j.item(), loss2.item(), loss3.item(), cur_lr)
        log_txt.write(log_str)
        log_txt.flush()
        print(step, 'loss', loss.item(),loss1.item(), 'loss_j: %.6f'%(loss_j.item()), 'loss_d: %.6f'%(loss2.item()), '%.6f'%(loss3.item()), '%.4f'%(cur_lr), d_weight)

