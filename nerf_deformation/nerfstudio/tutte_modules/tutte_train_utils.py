
import time
import os 
# import cv2 
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
# from cholespy import CholeskySolverF, MatrixType

import igl
# import pymesh
from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
import torch_sparse 
import torch_cluster

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import NearestNeighbors

import matplotlib 
 

def distortion_loss(distortion, dim=3, weighted=False, print_loss=False, return_i=False):
    # [N, 2,2]
    d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
    loss_i = torch.square(d - torch.eye(dim).unsqueeze(0).repeat(d.shape[0], 1,1)).mean(-1).mean(-1)
    
    if weighted:
        mask = loss_i > 0.01 
        mask2 = loss_i > 0.02 
        loss = 2 * loss_i[mask].mean() + loss_i.mean() + 5 * loss_i[mask2].mean() 
    else:
        loss = loss_i.mean() 
    if return_i:
        return loss, loss_i 
    else:
        return loss

def l2_loss(pred_points, gt_points, weighted=True, return_i=False):
    # [N, 2,2]
    loss_i = torch.square(pred_points[0] - gt_points[0]).mean(-1)
    
    if weighted:
        mask = loss_i > 0.0001
        if mask.sum()>0:
            loss = 3 * loss_i[mask].mean() + loss_i.mean() 
        else:
            loss = loss_i.mean() 
    else:
        loss = loss_i.mean() 
    if return_i:
        return loss, loss_i 
    else:
        return loss 
    


def jacobian_loss(pred_points, target_J, grad):
    pred_J = _multiply_sparse_2d_by_dense_3d(grad, pred_points).type_as(pred_points)
    pred_J = pred_J.view(pred_points.shape[0], -1, 3,3).transpose(2,3)
    loss = torch.square(pred_J - target_J).mean() 
    return loss 

def vol_jacob_loss(jacob, gt_transoformation):
    # # jacob: [N, 3,3]
    # loss_j_i = torch.square(jacob - gt_transoformation.unsqueeze(0)).mean(-1).mean(-1)
    # weights = torch.ones(loss_j_i.shape[0])
    # weights[torch.nonzero((loss_j_i>0.01), as_tuple=True)[0]] = 2 
    # weights[torch.nonzero((loss_j_i>0.04), as_tuple=True)[0]] = 4
    # weights[torch.nonzero((loss_j_i>0.08), as_tuple=True)[0]] = 8 
    # loss_j = (loss_j_i * weights).mean() 
    # print('loss_j', loss_j_i.max(), loss_j_i.min(), loss_j_i.mean())
    loss_j = torch.square(jacob - gt_transoformation.unsqueeze(0)).mean() 
    return loss_j 
