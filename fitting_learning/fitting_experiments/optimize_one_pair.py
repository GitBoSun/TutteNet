import time
import os
import sys
# import cv2
import random
import trimesh
import numpy as np
import numpy.linalg
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt
from psbody.mesh import Mesh

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

import matplotlib

import random
random.seed(11)


num_layer = 8
N =  11

model_name = 'fitting_n%dl%d'%( N, num_layer)
out_path = './fitting_train/%s'%(model_name)

def write_ply_color(points, dis, out_filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
    cmap = matplotlib.cm.get_cmap('bwr')

    fout = open(out_filename, 'w')
    ### Write header here
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n" % N)
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("end_header\n")
    print(dis.max(), dis.min(), dis.mean())
    for i in range(N):
        c = cmap(dis[i])
        c = [int(x*255) for x in c]

        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2], \
                                            c[0], c[1], c[2] ))

    fout.close()

USE_TORCH_SPARSE = True
class SparseMat:
    '''
    Sparse matrix object represented in the COO format
    Refacto : consider killing this object, byproduct of torch_sparse instead of torch.sparse (new feature)
    '''

    @staticmethod
    def from_M(M,ttype):
        return SparseMat(M[0],M[1],M[2],M[3],ttype)

    @staticmethod
    def from_coo(coo,ttype):
        inds = numpy.vstack((coo.row,coo.col))
        return SparseMat(inds,coo.data,coo.shape[0],coo.shape[1],ttype)

    def __init__(self,inds,vals,n,m,ttype):
        self.n = n
        self.m = m
        self.vals = vals
        self.inds = inds
        assert(inds.shape[0] == 2)
        assert(inds.shape[1] == vals.shape[0])
        assert(np.max(inds[0,:]) <= n)
        assert(np.max(inds[1,:] <= m))
        #TODO figure out how to extract the I,J,V,m,n from this, then load a COO mat directly from npz
        #self.coo_mat = coo_matrix((cupy.array(self.vals), (cupy.array(self.inds[0,:]), cupy.array(self.inds[1,:]))))
        self.vals = torch.from_numpy(self.vals).type(ttype).contiguous()
        self.inds = torch.from_numpy(self.inds).type(torch.int64).contiguous()

    def to_coo(self):
        return coo_matrix((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def to_csc(self):
        return sp_csc((self.vals, (self.inds[0,:], self.inds[1,:])), shape = (self.n, self.m))

    def to(self,device):
        self.vals = self.vals.to(device)
        self.inds = self.inds.to(device)
        return self

    def pin_memory(self):
        return
        # self.vals.pin_memory()
        # self.inds.pin_memory()

    def multiply_with_dense(self,dense):
        if USE_TORCH_SPARSE:
            res = torch_sparse.spmm(self.inds,self.vals, self.n, self.m, dense)
            # 1000 for loop on the above line takes 0.13 sec. Fast but annoying to have this dependency
        else:
            # Somehow this is not implemented for now?
            # res = torch.smm(torch.sparse_coo_tensor(self.inds,self.vals) , (dense.float())).to_dense().to(dense.device)
            # 1000 for loop on the above line takes 10 sec on the CPU. It is not implemented on gpu yet Slower but no dependency
            if self.vals.device.type == 'cpu':
                tensor_zero_hack  = torch.FloatTensor([0]).double() # This line was somehow responsible for a nasty NAN bug
            else:
                tensor_zero_hack  =  torch.cuda.FloatTensor([0]).to(dense.get_device()).double()
            # beware with addmm, it is experimental and gave me a NaN bug!
            res = torch.sparse.addmm(tensor_zero_hack, torch.sparse_coo_tensor(self.inds.double(),self.vals.double()) , (dense.double())).type_as(self.vals)
            # 1000 for loop on the above line takes 0.77 sec. Slower but no dependency
        return res.contiguous()

def _convert_sparse_igl_grad_to_our_convention(input):
    '''
    The grad operator computed from igl.grad() results in a matrix of shape (3*#tri x #verts).
    It is packed such that all the x-coordinates are placed first, followed by y and z. As shown below

    ----------           ----------
    | x1 ...             | x1 ...
    | x2 ...             | y1 ...
    | x3 ...             | z1 ...
    | .                  | .
    | .                  | .
    | y1 ...             | x2 ...
    | y2 ...      ---->  | y2 ...
    | y3 ...             | z2 ...
    | .                  | .
    | .                  | .
    | z1 ...             | x3 ...
    | z2 ...             | y3 ...
    | z3 ...             | z3 ...
    | .                  | .
    | .                  | .
    ----------           ----------

    Note that this functionality cannot be computed trivially if because igl.grad() is a sparse tensor and as such
    slicing is not well defined for sparse matrices. the following code performs the above conversion and returns a
    torch.sparse tensor.
    Set check to True to verify the results by converting the matrices to dense and comparing it.
    '''
    assert type(input) == sp_csc, 'Input should be a scipy csc sparse matrix'
    T = input.tocoo()

    r_c_data = np.hstack((T.row[..., np.newaxis], T.col[..., np.newaxis],
                          T.data[..., np.newaxis]))  # horizontally stack row, col and data arrays
    r_c_data = r_c_data[r_c_data[:, 0].argsort()]  # sort along the row column

    # Separate out x, y and z blocks
    '''
    Note that for the grad operator there are exactly 3 non zero elements in a row
    '''
    L = T.shape[0]
    Tx = r_c_data[:L, :]
    Ty = r_c_data[L:2 * L, :]
    Tz = r_c_data[2 * L:3 * L, :]

    # align the y,z rows with x so that they too start from 0
    Ty[:, 0] -= Ty[0, 0]
    Tz[:, 0] -= Tz[0, 0]

    # 'strech' the x,y,z rows so that they can be interleaved.
    Tx[:, 0] *= 3
    Ty[:, 0] *= 3
    Tz[:, 0] *= 3

    # interleave the y,z into x
    Ty[:, 0] += 1
    Tz[:, 0] += 2

    Tc = np.zeros((input.shape[0] * 3, 3))
    Tc[::3] = Tx
    Tc[1::3] = Ty
    Tc[2::3] = Tz

    indices = Tc[:, :-1].astype(int)
    data = Tc[:, -1]

    return (indices.T, data, input.shape[0], input.shape[1])
def _multiply_sparse_2d_by_dense_3d(mat, B):
    ret = []
    for i in range(B.shape[0]):
        C = mat.multiply_with_dense(B[i, ...])
        ret.append(C)
    ret = torch.stack(tuple(ret))
    return ret

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
        self.mesh_resolution = mesh['mesh_resolution']
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
        W_var = torch.empty(self.batch_size, n_edges)
        torch.nn.init.uniform_(W_var)
        self.W_var = nn.Parameter(W_var)

        angle_var = torch.empty(self.batch_size, len(bound_verts))
        torch.nn.init.uniform_(angle_var)
        self.angle_var = nn.Parameter(angle_var)

    def forward(self, input_points, inverse=False):
        if inverse:
            return self.forward_inverse(input_points)
        
        input_points_np = input_points.detach().numpy()
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]

        input_nodes = self.find_simplex(input_points.view(batch_size*N_points, 2)).numpy()
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)

        input_areas = self.get_areas(input_points, input_nodes) # [b, N, 3]
        new_vertices = self.tutte_embedding_sparse()
        pred_points = self.get_tutte_from_triangle(new_vertices, input_nodes, input_areas)

        distortions = self.compute_distortion(input_nodes, self.vertices, new_vertices)
        return pred_points, new_vertices, distortions

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
        W_var = torch.sigmoid(self.W_var)*(1-0.4) + 0.2

        batch_size = W_var.shape[0]
        angle_var = torch.sigmoid(self.angle_var)*(1-0.4) + 0.2

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
        return distortion

    def find_simplex(self, points,):
        N = self.mesh_resolution
        points = points / 2 + 0.5
        n_points = points.shape[0]
        interval = float(1/(N-1))
        y_ids = points[:, 0]//interval
        x_ids = (1-points[:, 1])//interval

        out_triangle_verts = torch.zeros((n_points, 3)).long()

        top_left_ids = (x_ids * N + y_ids).long()
        top_right_ids = (x_ids * N + y_ids  + 1 ).long()
        bottom_left_ids = ((x_ids+1) * N + y_ids).long()
        bottom_right_ids = ((x_ids+1) * N + y_ids + 1 ).long()
        center_ids = (x_ids * (N-1) + y_ids + N*N ).long()

        vertices  = self.vertices / 2 + 0.5
        center_verts = vertices[center_ids]

        difference = points - center_verts
        tangent = difference[:,1] /(difference[:,0] + 1e-5)
        mask1 = torch.nonzero((difference[:,0]>=0) & (tangent <=1) & (tangent >-1), as_tuple=True)[0]
        mask2 = torch.nonzero((difference[:,1]>=0) & ((tangent >1) | (tangent <=-1)), as_tuple=True)[0]
        mask3 = torch.nonzero((difference[:,0]<0) & (tangent <=1) & (tangent >-1), as_tuple=True)[0]
        mask4 = torch.nonzero((difference[:,1]<0) & ((tangent >1) | (tangent <=-1)), as_tuple=True)[0]

        out_triangle_verts[:,0] = center_ids

        out_triangle_verts[mask1,1] =  top_right_ids[mask1]
        out_triangle_verts[mask1,2] =  bottom_right_ids[mask1]

        out_triangle_verts[mask2,1] =  top_left_ids[mask2]
        out_triangle_verts[mask2,2] =  top_right_ids[mask2]

        out_triangle_verts[mask3,1] =  bottom_left_ids[mask3]
        out_triangle_verts[mask3,2] =  top_left_ids[mask3]

        out_triangle_verts[mask4,1] =  bottom_right_ids[mask4]
        out_triangle_verts[mask4,2] =  bottom_left_ids[mask4]

        return out_triangle_verts

class TutteTriplane(nn.Module):
    def __init__(self, mesh, lap_dict, radius=1, prefix=''):
        super(TutteTriplane, self).__init__()
        self.tutte_layer1 = TutteLayer(mesh, lap_dict, radius)
        self.tutte_layer2 = TutteLayer(mesh, lap_dict, radius)
        self.tutte_layer3 = TutteLayer(mesh, lap_dict, radius)
        self.prefix = prefix

    def forward(self, input_points, inverse=False):
        if inverse:
            return self.forward_inverse(input_points)

        prefix = self.prefix
        new_points = input_points

        cur_points1 = new_points[:,:,0:2].clone()
        pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1, inverse=inverse)
        new_points1 = torch.cat((pred_points1, new_points[:,:,2].unsqueeze(2)), dim=2)

        cur_points2 = torch.cat((new_points1[:,:,0].unsqueeze(2), new_points1[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2, inverse=inverse)
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points1[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2)

        cur_points3 = new_points2[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3, inverse=inverse)
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

    def forward_inverse(self, input_points, ):
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

        self.tutte_layers = nn.ModuleList([TutteTriplane(mesh, self.lap_dict, radius,  prefix='L%d_'%(i) ) \
                for i in range(num_layer)])
    
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

if __name__=="__main__":
    temp_mesh = Mesh(filename = './meshes/temp_pose.obj')
    vert0 = temp_mesh.vertices

    mesh1 = Mesh(filename = './meshes/1100.obj')
    scale_factor = 1.4
    vert1 = mesh1.vertices
    mesh2 = Mesh(filename = './meshes/2900.obj')

    vert2 = mesh2.vertices
    print(vert1.shape, vert2.shape)
    alpha = np.pi/2
    rot_matrix = np.array([ [1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
    vert0 = np.matmul(vert0,rot_matrix )
    vert1 = np.matmul(vert1,rot_matrix )
    vert2 = np.matmul(vert2,rot_matrix )

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
    print(vert1.min(0),vert1.max(0))

    vert2 = vert2 - min1
    vert2 = vert2 / (max1[2] - min1_1[2])
    vert2 = (vert2 - (max1_2 -min1_2)/2)*scale_factor
    print(vert2.min(0),vert2.max(0))

    handle_points = vert1
    handle_target = vert2

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    log_txt = open(os.path.join(out_path, 'log.txt'), 'a')

    shape_points = torch.from_numpy(vert1).float().unsqueeze(0)
    input_points = torch.from_numpy(handle_points).float().unsqueeze(0)
    target_points = torch.from_numpy(handle_target).float().unsqueeze(0)
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
    mesh_input['vertices'] = torch.from_numpy(vertices).float()
    mesh_input['edges'] = torch.from_numpy(edges).long()
    mesh_input['faces'] = faces
    mesh_input['mesh_resolution'] = N

    def distortion_loss(distortion, dim=3, weighted=False, print_loss=False, return_i=False):
        # [N, 2,2]
        d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
        loss_i = torch.square(d - torch.eye(dim).unsqueeze(0).repeat(d.shape[0], 1,1)).mean(-1).mean(-1)

        if weighted:
            mask = loss_i>0.01
            mask2 = loss_i>0.02
            loss = 2 * loss_i[mask].mean() + loss_i.mean() + 5 * loss_i[mask2].mean()
        else:
            loss = loss_i.mean()
        if return_i:
            return loss, loss_i
        else:
            return loss

    grad = igl.grad(vert0, mesh1.f.astype(np.int32))
    igl_grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad.tocsc()),torch.float64)
    target_grad = _multiply_sparse_2d_by_dense_3d(igl_grad, target_points).type_as(target_points)
    target_grad = target_grad.view(target_points.shape[0], -1, 3,3).transpose(2,3)

    def jacobian_loss(pred_points, target_J):
        pred_J = _multiply_sparse_2d_by_dense_3d(igl_grad, pred_points).type_as(pred_points)
        pred_J = pred_J.view(pred_points.shape[0], -1, 3,3).transpose(2,3)
        loss = torch.square(pred_J - target_J).mean()
        return loss

    tutte_model = TutteModel(mesh_input, bound_verts, num_layer=num_layer)
    print('model size', sum([m.numel() for m in tutte_model.parameters()]))

    lr = 0.02 # 0.02
    optim =  torch.optim.Adam(tutte_model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=5000)
    # scheduler = lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.01, total_iters=4000)
    creteria = torch.nn.MSELoss()

    d_weight = 0.00
    r_weight = 0.00
    d_weight_or = d_weight
    r_weight_or = r_weight
    j_weight = 0.1
    weighted = False

    prefix = 'triplane_'
    t1 = time.time()
    start_step = 0

    num_input = input_points.shape[1]
    num_shape = shape_points.shape[1]

    model_dir = 'smpl_model'
    model_path = model_dir +'/%s.pt'%(model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        tutte_model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
        print('load model from step', start_step)

    for step in range(start_step,10000):
        input_points.requires_grad=True

        optim.zero_grad()
        forward_points = input_points
        return_dict = tutte_model(forward_points)
        return_dict1 = return_dict

        pred_points = return_dict['pred_points']
        loss1 = creteria(pred_points, target_points)
        loss_J = jacobian_loss(pred_points, target_grad)

        loss2, loss2_i = distortion_loss(return_dict1['total_distortion'],dim=3, weighted=weighted, return_i=True)

        loss3 = 0.0
        for i in range(num_layer):
            loss3 += distortion_loss(return_dict1['L%d_distortion1'%(i)],dim=2,)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion2'%(i)],dim=2,)/3
            loss3 += distortion_loss(return_dict1['L%d_distortion3'%(i)],dim=2,)/3
        loss3 = loss3 / num_layer
        loss = loss1 + d_weight*loss2 + r_weight*loss3 + j_weight * loss_J

        loss.backward()
        optim.step()
        scheduler.step()

        if step%200==0:
            total_norm = 0
            parameters = [p for p in tutte_model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            log_str = '[grad norm]: %.7f\n'%(total_norm)
            log_txt.write(log_str)
            log_txt.flush()
            print('grad norm', total_norm, )

            pred_points_np = return_dict1['pred_points'] .detach().cpu().numpy()[0]
            alpha = -np.pi/2
            rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
            pred_points_np = np.matmul(pred_points_np,rot_matrix )

            loss2_i = torch.sqrt(torch.square(pred_points[0] - target_points[0]).sum(-1))
            write_ply_color(pred_points_np, torch.clamp(loss2_i, 0, 0.01).detach().numpy()/0.01, os.path.join(out_path, prefix+'_%d_pred_%.4f_color.ply'%(step, loss2.item())))
            mesh3 = trimesh.Trimesh(vertices=pred_points_np, faces=temp_mesh.f)
            mesh3.export(os.path.join(out_path, prefix+'_%d_pred_%.4f.ply'%(step, loss2.item())))
            mesh3 = trimesh.Trimesh(vertices=np.matmul(vert1,rot_matrix ), faces=temp_mesh.f)
            mesh3.export(os.path.join(out_path, prefix+'_input.ply'))

            mesh3 = trimesh.Trimesh(vertices=np.matmul(vert2,rot_matrix ), faces=temp_mesh.f)
            mesh3.write_ply(os.path.join(out_path, prefix+'_target.ply'))

            torch.save({
                'step': step,
                'model_state_dict': tutte_model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss,
                }, model_path)

        if step%20==0:
            cur_lr = optim.param_groups[-1]['lr']
            log_str = 'step: %d, loss: %.8f, loss_l2: %.9f, loss_d: %.4f, loss_r: %.4f, lr: %.4f\n'%(\
                step, loss.item(),loss1.item(),loss2.item(), loss3.item(), cur_lr)
            log_txt.write(log_str)

            print(step, 'loss', loss.item(),loss1.item(), loss_J.item(), '%.6f'%(loss2.item()), '%.6f'%(loss3.item()), '%.4f'%(cur_lr), model_name)


    t2 = time.time()
    print('total training time', t2-t1)
