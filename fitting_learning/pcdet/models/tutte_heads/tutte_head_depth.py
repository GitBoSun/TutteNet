import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter
from torch_sparse_solve import solve

import random
import trimesh
import numpy as np
from psbody.mesh import Mesh
import numpy.linalg
from scipy.spatial import Delaunay

import igl
from scipy.sparse import diags,coo_matrix
from scipy.sparse import csc_matrix as sp_csc
import torch_sparse 

from ...utils import loss_utils


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
        self.vals = torch.from_numpy(self.vals).type(ttype).contiguous().cuda()
        self.inds = torch.from_numpy(self.inds).type(torch.int64).contiguous().cuda()

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
            
            # tensor_zero_hack  =  torch.cuda.FloatTensor([0]).to(dense.get_device()).double()
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

class TutteLayerDepth(nn.Module):
    def __init__(self, mesh, lap_dict, model_cfg, runtime_cfg, radius=1,):
        super(TutteLayerDepth, self).__init__()
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.radius = radius
        self.epsilon = model_cfg.get("EPSILON", 0.01)
        self.epsilon_angle = model_cfg.get("EPSILON_ANGLE", 0.01)
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

    def forward(self, input_points, var_pred, depth, inverse=False):

        input_points_np = input_points.clone().detach().cpu().numpy()
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        input_points_np = np.reshape(input_points_np, (batch_size*N_points, 2))
        face_ids = Delaunay.find_simplex(self.tri, input_points_np.astype(np.float64), bruteforce=True)
        input_nodes = self.faces[face_ids]
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)

        input_areas = self.get_areas(input_points, input_nodes) # [b, N, 3]
        
        new_vertices1 = self.tutte_embedding_sparse(var_pred[:,:,0])
        pred_points1 = self.get_tutte_from_triangle(new_vertices1, input_nodes, input_areas)
        
        new_vertices2 = self.tutte_embedding_sparse(var_pred[:,:,1])
        pred_points2 = self.get_tutte_from_triangle(new_vertices2, input_nodes, input_areas)
        
        depth = depth.unsqueeze(2)
        pred_points = pred_points1 * (1-depth)/2 +  pred_points2 * (depth+1)/2 
        
        d_faces = np.reshape(self.faces[face_ids], (batch_size, N_points, 3)).astype(np.int32)
        distortions1 = self.compute_distortion(d_faces, self.vertices, new_vertices1)
        distortions2 = self.compute_distortion(d_faces, self.vertices, new_vertices2)
        distortions =  distortions1 * (1-depth)/2 +  distortions2 * (depth+1)/2 
        # return_dict = {'new_vertices': new_vertices1, 'distortion': distortions}
        return pred_points, new_vertices1, distortions 

    def forward_inverse(self, input_points, var_pred):
         
        new_vertices = self.tutte_embedding_sparse(var_pred) 
        
        input_points_np = input_points.clone().detach().cpu().numpy() 
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        
        face_ids = np.zeros((batch_size, N_points), np.int32)
        for i in range(batch_size):
            tri = Delaunay(new_vertices[i].detach().cpu().numpy())
            tri.simplices = np.array(self.faces, np.int32)
            input_points_np_i = input_points_np[i] 
            face_ids_i = Delaunay.find_simplex(tri, input_points_np_i.astype(np.float64), bruteforce=True) 
            face_ids[i] = face_ids_i 
            
        input_nodes = self.faces[face_ids.reshape(-1)] 
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)
        
        input_areas = self.get_areas_inverse(input_points, input_nodes, new_vertices) # [b, N, 3]
        pred_points = self.get_tutte_from_triangle(self.vertices.unsqueeze(0).repeat(batch_size, 1, 1), input_nodes, input_areas)
        distortions = torch.zeros(batch_size* N_points, 2,2).cuda() 
        
        return pred_points, new_vertices, distortions 

    def tutte_embedding_sparse(self, var_pred):
        """
        Args:
            W_var: [b, n_edge]
            angle_var: [b, n_bound]
        """
        W_var = var_pred[:, :self.num_edges]
        angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)]
        
        # if self.fix_bound:
        #     angle_var = None 
        # elif not self.rotate_angle:
        #     angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)]
        # else:
        #     angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)+1]

        angle_var = angle_var / self.angle_scale
        angle_var = torch.sigmoid(angle_var) * (1-2*self.epsilon_angle) + self.epsilon_angle
        
        if self.inverse_w:
            W_var = torch.abs(W_var)
            W_var = 1/(W_var + 1)
            # W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
            # W_var = 2* (W_var - 0.5 )
        elif self.use_sigmoid:
            W_var = W_var/self.w_scale
            W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
        
        n_vert = self.vertices.shape[0]
        batch_size = W_var.shape[0]
        
        if angle_var is None:
            angle_init = 2 * torch.pi * torch.arange(len(self.bound_verts)).float().cuda()/len(self.bound_verts)
            angle_init = angle_init.unsqueeze(0).repeat(batch_size, 1)
            
        elif self.rotate_angle:
            tmp_angle =  angle_var[:, 1:].clone()
            new_angle = tmp_angle / tmp_angle.sum(1).unsqueeze(1)
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
    
    def get_areas_inverse(self, points, tri_nodes, vert):
        """

        Args:
            points: [b, N, 3]
            tri_nodes: [b, N, 3]
        """
        areas = torch.zeros(tri_nodes.shape).cuda() 
        tri_nodes_new = tri_nodes.copy()
         
        batch_size, n_vert, _ = vert.shape 
        n_points = points.shape[1]
        
        vertices = vert.view(batch_size * n_vert, 2)
        for i in range(batch_size):
            tri_nodes_new[i] += n_vert * i  
            
        tri_nodes_new = np.reshape(tri_nodes_new, (batch_size * n_points, 3))
    
        loc = points
        init_A = vertices[tri_nodes_new[:,0].astype(np.int32), :].view(batch_size, n_points, 2)
        init_B = vertices[tri_nodes_new[:,1].astype(np.int32), :].view(batch_size, n_points, 2)
        init_C = vertices[tri_nodes_new[:,2].astype(np.int32), :].view(batch_size, n_points, 2)
    
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
        tri_nodes_new = tri_nodes.copy() 
        for i in range(batch_size):
            tri_nodes_new[i] += n_vert * i
        tri_nodes_new = np.reshape(tri_nodes_new, (batch_size * n_points, 3))
              
        area_A = areas[:, 0].unsqueeze(1)
        area_B = areas[:, 1].unsqueeze(1)
        area_C = areas[:, 2].unsqueeze(1)
        total_area = area_A + area_B + area_C
        new_A = pos[tri_nodes_new[:,0].astype(np.int32)] # [b, N, 2]
        new_B = pos[tri_nodes_new[:,1].astype(np.int32)]
        new_C = pos[tri_nodes_new[:,2].astype(np.int32)]
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

class TutteTriplane(nn.Module):
    def __init__(self, mesh, lap_dict,  model_cfg, runtime_cfg, radius=1, prefix=''):
        super(TutteTriplane, self).__init__()
        self.tutte_layer1 = TutteLayer(mesh, lap_dict,  model_cfg, runtime_cfg, radius)
        self.tutte_layer2 = TutteLayer(mesh, lap_dict,  model_cfg, runtime_cfg, radius)
        self.tutte_layer3 = TutteLayer(mesh, lap_dict,  model_cfg, runtime_cfg, radius)
        self.prefix = prefix 
    
    def forward(self, input_points, var_pred, inverse=False):
        if inverse:
            return self.forward_inverse(input_points, var_pred)
        
        prefix = self.prefix
        new_points = input_points
    
        cur_points1 = new_points[:,:,0:2].clone() 
        pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1, var_pred[:,:,0])
        new_points1 = torch.cat((pred_points1, new_points[:,:,2].unsqueeze(2)), dim=2)
        
        cur_points2 = torch.cat((new_points1[:,:,0].unsqueeze(2), new_points1[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2,var_pred[:,:,1] )
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points1[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2) 
        
        cur_points3 = new_points2[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3, var_pred[:,:,2])
        new_points3 = torch.cat((new_points2[:,:,0].unsqueeze(2), pred_points3), dim=2)
        
        total_distortion = self.compute_distortion(distortions1, distortions2,distortions3)
        
        return_dict = {} 
        return_dict[prefix+'pred_points1'] = new_points1 
        return_dict[prefix+'new_01_vertices1'] = new_vertices1
        return_dict[prefix+'pred_01_points1'] = pred_points1
        return_dict[prefix+'distortion1'] = distortions1 
        
        return_dict[prefix+'pred_points2'] = new_points2
        return_dict[prefix+'new_12_vertices2'] = new_vertices2 
        return_dict[prefix+'pred_12_points2'] = pred_points2
        return_dict[prefix+'distortion2'] = distortions2
        
        return_dict[prefix+'pred_points3'] = new_points3
        return_dict[prefix+'new_02_vertices2'] = new_vertices3 
        return_dict[prefix+'pred_02_points2'] = pred_points3
        return_dict[prefix+'distortion3'] = distortions3 
        
        return_dict[prefix+'distortion'] = total_distortion 
        
        return new_points3, return_dict
    
    def forward_inverse(self, input_points,var_pred):
        prefix = self.prefix
        new_points = input_points
        
        cur_points3 = new_points[:,:,1:3].clone()
        pred_points3, new_vertices3, distortions3 = self.tutte_layer3(cur_points3, var_pred[:,:,2], inverse=True)
        new_points3 = torch.cat((new_points[:,:,0].unsqueeze(2), pred_points3), dim=2)
        
        cur_points2 = torch.cat((new_points3[:,:,0].unsqueeze(2), new_points3[:,:,2].unsqueeze(2)), dim=2)
        pred_points2, new_vertices2, distortions2 = self.tutte_layer2(cur_points2, var_pred[:,:,1], inverse=True)
        new_points2 = torch.cat((pred_points2[:,:,0].unsqueeze(2), new_points3[:,:,1].unsqueeze(2), pred_points2[:,:,1].unsqueeze(2)), dim=2) 
        
        cur_points1 = new_points2[:,:,0:2].clone() 
        pred_points1, new_vertices1, distortions1 = self.tutte_layer1(cur_points1, var_pred[:,:,0], inverse=True)
        new_points1 = torch.cat((pred_points1, new_points2[:,:,2].unsqueeze(2)), dim=2)
        
        return_dict = {}         
        return new_points1, return_dict
    
    
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
    def __init__(self, mesh, lap_dict, normal_matrices,  model_cfg, runtime_cfg,radius=1, prefix='',):
        super(TutteVaryingNormal, self).__init__()
        
        self.normal_matrices = normal_matrices
        self.num_layer = self.normal_matrices.shape[0]
        
        self.tutte_layers = nn.ModuleList([TutteLayerDepth(mesh, lap_dict,  model_cfg, runtime_cfg, radius) \
                for i in range(self.num_layer)])
        self.prefix = prefix 
    
    def forward(self, input_points, var_pred, inverse=False):
        if inverse:
            return self.forward_inverse(input_points, var_pred)
        
        prefix = self.prefix
        new_points = input_points.float()
        return_dict = {} 
        total_distortion = None 
        
        for i in range(self.num_layer):
            rot_points = torch.matmul(self.normal_matrices[i],new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            cur_points = rot_points[:,:,0:2].clone() 
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points,  var_pred[:,:,i], depth=cur_points[:,:,2])
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

    def forward_inverse(self, input_points, var_pred):
        prefix = self.prefix
        new_points = input_points
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layer-1, -1, -1):
            rot_points = torch.matmul(self.normal_matrices[i],new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            cur_points = rot_points[:,:,0:2].clone() 
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points,  var_pred[:,:,i], inverse=True)
            new_points = torch.cat((pred_points, rot_points[:,:,2].unsqueeze(2)), dim=2) 
            new_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
        return new_points, return_dict

class TuttePredictingNormal(nn.Module):
    def __init__(self, mesh, lap_dict,  model_cfg, runtime_cfg,radius=1, prefix='',):
        super(TuttePredictingNormal, self).__init__()
        # self.num_layer = runtime_cfg['num_layers']
        self.num_inner_layer = runtime_cfg["num_inner_layers"]
        self.tutte_layers = nn.ModuleList([TutteLayer(mesh, lap_dict,  model_cfg, runtime_cfg, radius) \
                for i in range(self.num_inner_layer)])
        self.prefix = prefix 
    
    def forward(self, input_points, var_pred, normal_pred):
        prefix = self.prefix
        new_points = input_points.float()
        return_dict = {} 
        total_distortion = None 
        batch_size = var_pred.shape[0]
        normal_pred = normal_pred.view(batch_size*self.num_inner_layer, 3)
        self.normal_matrices = self.rotation_matrix_from_vectors(normal_pred) # [B*n, 3,3]
        self.normal_matrices = self.normal_matrices.view(batch_size, self.num_inner_layer, 3,3)
                
        for i in range(self.num_inner_layer):
            # [B, 3,3] [B, 3, M]
            rot_points = torch.matmul(self.normal_matrices[:, i], new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            cur_points = rot_points[:,:,0:2].clone() 
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points,  var_pred[:,:,i])
            new_points = torch.cat((pred_points, rot_points[:,:,2].unsqueeze(2)), dim=2) 
            
            new_points = torch.matmul(self.normal_matrices[:, i].transpose(1,2), \
                new_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            cur_distortion = torch.eye(3).unsqueeze(0).repeat(distortions.shape[0],1,1).cuda()
            cur_distortion[:, :2,:2] = distortions 
            cur_distortion = cur_distortion.view(batch_size, -1, 3,3,)
            cur_distortion = torch.matmul(self.normal_matrices[:, i].transpose(1,2).unsqueeze(1), cur_distortion)
            cur_distortion = torch.matmul(cur_distortion, self.normal_matrices[:, i].unsqueeze(1), ) 
            cur_distortion = cur_distortion.view(-1, 3,3)
            
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
    
    def rotation_matrix_from_vectors(self, vec1, ):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        # a = (vec1 / torch.norm(vec1)).view(3)
        # b = (vec2 / torch.norm(vec2)).view(3)
        batch_size = vec1.shape[0]
        a = vec1
        b = torch.zeros(batch_size, 3).cuda().float() 
        b[:,2] = 1.0 
        v = torch.cross(a, b) # [B, 3,]
        if torch.any(v): #if not all zeros then 
            # c = torch.dot(a, b)
            c = torch.sum(a*b, dim=1).unsqueeze(1).unsqueeze(2)
            s = torch.norm(v, dim=1).unsqueeze(1).unsqueeze(2)
            
            kmat = torch.zeros((batch_size, 3,3)).cuda() 
            kmat[:, 0, 1] = -v[:,2] 
            kmat[:, 0, 2] = v[:,1] 
            kmat[:, 1, 0] = v[:,2]
            kmat[:, 1, 2] = -v[:,0] 
            kmat[:, 2, 0] = -v[:,1] 
            kmat[:, 2, 1] = v[:,0] 
            # kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return torch.eye(3).unsqueeze(0).repeat(batch_size, 1,1).cuda() + \
                kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2))
        else:
            return torch.eye(3).unsqueeze(0).repeat(batch_size, 1,1).cuda() #cross of all zeros only occurs on identical directions
        
class TutteHead3D_Distortion(nn.Module):

    def __init__(self, mesh, bound_verts, runtime_cfg, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_layers = runtime_cfg["num_layers"]
        
        self.gather_middle_results = model_cfg.get("GATHER_MIDDLE_RESULTS", False)
        
        self.area_loss_weight = model_cfg.get("AREA_LOSS_WEIGHT", False)
        self.add_lap_loss = model_cfg.get("ADD_LAP_LOSS", False)
        self.lap_loss_weight = model_cfg.get("LAP_LOSS_WEIGHT", 0)
        self.r_weight = model_cfg.get("LAYER_DISTORTION_WEIGHT", 0)
        self.add_layer_distortion_loss = model_cfg.get("ADD_LAYER_DISTORTION_LOSS", False)
        self.d_weight = model_cfg.get("DISTORTION_WEIGHT", 0)
        self.add_distortion_loss = model_cfg.get("ADD_DISTORTION_LOSS", False)
        self.j_weight = model_cfg.get("JACOBIAN_WEIGHT", 0)
        self.add_jacobian_loss = model_cfg.get("ADD_JACOBIAN_LOSS", False)
        
        self.use_full = model_cfg.get("USE_FULL", False) 
        self.use_triplane = model_cfg.get("USE_TRIPLANE", True) 
        self.rotate_triplane = model_cfg.get("ROTATE_TRIPLANE", False) 
        self.predict_normal = runtime_cfg["predict_normal"] 
        self.use_shape = runtime_cfg["use_shape"] 
        
        self.use_two_directions = model_cfg.get("USE_TWO_DIRECTIONS", False)  
        self.m_weight = model_cfg.get("MIDDLE_WEIGHT_WEIGHT", 0)
        self.compute_inverse = model_cfg.get("COMPUTE_INVERSE", True)  

        self.normals = model_cfg.get("NORMALS", [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                                 [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]) 
        self.normals = np.array(self.normals, np.float32)
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        if not self.use_triplane and not self.predict_normal:
            print("normals", self.normals)
            self.num_inner_layer = self.normals.shape[0]
        elif self.predict_normal:
            self.num_inner_layer = runtime_cfg["num_inner_layers"]
        else:
            self.num_inner_layer = 3
    
        self.mesh = mesh
        self.bound_verts = bound_verts
        self.input_radius = model_cfg.get("INPUT_RADIUS", 0.8 )

        self.temp_vert = runtime_cfg["temp_vert"]
        self.fps_ids = runtime_cfg["fps_ids"]
        self.temp_face = runtime_cfg["temp_face"]
        
        grad = igl.grad(self.temp_vert.cpu().numpy()[0], self.temp_face.astype(np.int32))
        self.igl_grad = SparseMat.from_M(_convert_sparse_igl_grad_to_our_convention(grad.tocsc()),torch.float64)
        
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
        
        # self.tutte_layers = nn.ModuleList([TutteLayer(mesh, self.lap_dict,  model_cfg, runtime_cfg, ) \
        #         for i in range(self.num_layers)])
        self.normal_matrices = np.zeros((self.normals.shape[0], 3,3))
        for i in range(self.normals.shape[0]):
            self.normal_matrices[i] = self.rotation_matrix_from_vectors(self.normals[i])
        self.normal_matrices = torch.from_numpy(self.normal_matrices).float().cuda().unsqueeze(1)
        
        if self.use_triplane:
            self.tutte_layers = nn.ModuleList([TutteTriplane(mesh, self.lap_dict, model_cfg, runtime_cfg,  prefix='l%d_'%(i)) \
                for i in range(self.num_layers)])
        elif self.predict_normal:
            self.tutte_layers = nn.ModuleList([TuttePredictingNormal(mesh, self.lap_dict, model_cfg, runtime_cfg,  prefix='l%d_'%(i) ) \
                for i in range(self.num_layers)]) 
        else:
            self.tutte_layers = nn.ModuleList([TutteVaryingNormal(mesh, self.lap_dict,self.normal_matrices, model_cfg, runtime_cfg,  prefix='l%d_'%(i) ) \
                for i in range(self.num_layers)]) 
        
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        
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
                for k in range(3):
                    w_var = var_pred[j, :edges.shape[1], k]
                    _, lap_value = torch_geometric.utils.get_laplacian(self.edges, w_var)
                    diag_values = lap_value[self.diag_ids] 
                    lap_loss += torch.square(diag_values - 1).mean() 
        lap_loss = lap_loss / (3*self.num_layers *var_pred.shape[0])
        return lap_loss
    
    def get_jacobian_loss(self,):
        
        target_points = self.forward_ret_dict['target_points']
        pred_points = self.forward_ret_dict['pred_points']
        
        target_J = _multiply_sparse_2d_by_dense_3d(self.igl_grad, target_points).type_as(target_points)
        target_J = target_J.view(target_points.shape[0], -1, 3,3).transpose(2,3)

        pred_J = _multiply_sparse_2d_by_dense_3d(self.igl_grad, pred_points).type_as(pred_points)
        pred_J = pred_J.view(pred_points.shape[0], -1, 3,3).transpose(2,3)
        loss = torch.square(pred_J - target_J).mean() 
        return loss 

    def get_distortion_loss(self, distortion, dim=2):
        # [N, 2,2]
        
        d = torch.matmul(distortion, torch.transpose(distortion, 1, 2))
        loss = torch.square(d - torch.eye(dim).cuda().unsqueeze(0).repeat(d.shape[0], 1,1)).mean()  
        return loss  
    
    def get_middle_loss(self):
        if 'pred_points_1' in self.forward_ret_dict.keys():
            target_points = self.forward_ret_dict['pred_points_1']
            pred_points = self.forward_ret_dict['pred_points_2']
            loss = torch.square(target_points-pred_points).mean()  
            return loss  
        else:
            return 0.0 
    
    def get_loss(self, tb_dict=None, prefix=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss, tb_dict_1 = self.get_regress_loss(prefix=prefix)
        tb_dict['reg_loss'] = point_loss.item()

        if self.add_lap_loss:
            lap_loss = self.get_laplacian_loss()
            point_loss += self.lap_loss_weight * lap_loss
            # print("lossssss", point_loss.item(), lap_loss.item())
            tb_dict['lap_loss'] = lap_loss.item()
        
        d_loss = 0.0 
        if self.add_distortion_loss:
            d_loss = self.get_distortion_loss(self.forward_ret_dict['total_distortion'], dim=3)
            d_loss = d_loss / self.num_layers  
            point_loss += self.d_weight * d_loss 
            tb_dict['d_loss'] = d_loss.item()
            
        r_loss = 0.0 
        if self.add_layer_distortion_loss:    
            for i in range(self.num_layers):
                for j in range(self.num_inner_layer):
                    r_loss += self.get_distortion_loss(self.forward_ret_dict['distortion%d_%d'%(i, j)],dim=2)/self.num_inner_layer
                r_loss = r_loss/self.num_layers
                
            tb_dict['r_loss'] = r_loss.item()
            point_loss += self.r_weight * r_loss 
        
        j_loss = 0.0 
        if self.add_jacobian_loss:
            j_loss = self.get_jacobian_loss()
            tb_dict['j_loss'] = j_loss.item()
            point_loss += self.j_weight * j_loss 
        
        m_loss = 0.0 
        if self.use_two_directions:
            m_loss = self.get_middle_loss()
            tb_dict['m_loss'] = m_loss.item()
            point_loss += self.m_weight * m_loss 
            
            
        return point_loss, tb_dict

    def forward(self, batch_dict, inverse=False):
        if self.use_two_directions:
            return self.forward_two_directions(batch_dict, inverse=inverse)
        
        batch_size =  batch_dict['batch_size']
        
        if self.use_shape:
            temp_points = batch_dict['temp_points']
        else:
            temp_points = self.temp_vert.repeat(batch_size, 1,1)
            
        shape_points = batch_dict['shape_points']
    
        if not self.use_full:
            input_points = temp_points[:, self.fps_ids, :]
            target_points = shape_points[:, self.fps_ids, :]
        else:
            input_points = temp_points
            target_points = shape_points
        
        # print(input_points.shape)
        num_points = temp_points.shape[1]
        ret_dict = {
            'pose_param': batch_dict['pose_param'],
            'target_points': target_points,
            # 'simplices': self.tutte_layers[0].tri.simplices,
            # 'vertices': self.mesh['vertices'],
            'batch_size': input_points.shape[0],
            'names': batch_dict['idx'],
            'input_points': input_points,
            'temp_points': temp_points, 
            'shape_points': shape_points, 
            'fps_ids': self.fps_ids, 
            'temp_face': self.temp_face, 
        }  

        total_distortion = None  
        pred_points = input_points.clone()
        pred_points1 = temp_points.clone() ###### 
        
        for i in range(self.num_layers):
            var_pred = batch_dict['var_pred%d'%(i)]
            var_pred = var_pred.view(batch_size, -1, self.num_inner_layer)
            if self.predict_normal:
                normal_pred = batch_dict['normal_pred'] 
                
            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i], pred_points.float().transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if self.predict_normal:
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, normal_pred)
            else:
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred)
            
            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                    pred_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if not self.use_full:   
                pred_points1, mid_dict1 = self.tutte_layers[i](pred_points1, var_pred)
            else:
                mid_dict1 = mid_dict
                pred_points1 = pred_points 
            
            # batch_dict.update(mid_dict)
            if 'l%d_distortion'%(i) in mid_dict1.keys():
                distortion = mid_dict1['l%d_distortion'%(i)]
                if self.rotate_triplane:
                    distortion = torch.matmul(self.normal_matrices[i].transpose(1,2), distortion)
                    distortion = torch.matmul(distortion, self.normal_matrices[i], ) 
            
                if total_distortion is None:
                    total_distortion = distortion 
                else: 
                    total_distortion = torch.matmul(distortion, total_distortion)
                batch_dict['distortion%d'%(i)] = distortion.clone()
                ret_dict.update({'distortion%d'%(i): distortion,}) 
                for j in range(self.num_inner_layer):
                    ret_dict.update({'distortion%d_%d'%(i, j): mid_dict1['l%d_distortion%d'%(i, j+1)],})
                    
            ret_dict.update({'var_pred%d'%(i): var_pred})
            
        # num_points = 69
        # total_distortion = total_distortion.view(batch_size, num_points, 3,3)
        if total_distortion is not None:
            ret_dict['total_distortion'] = total_distortion 
        
        if self.predict_normal:
            ret_dict['normal_pred'] = normal_pred  
            
        # def exp_func(x):
        #     pred_points = x.clone()
        #     for i in range(self.num_layers):
        #         var_pred = batch_dict['var_pred%d'%(i)]
        #         var_pred = var_pred.view(batch_size, -1, self.num_inner_layer)
        #         pred_points, _ = self.tutte_layers[i](pred_points, var_pred)
        #     return pred_points 
        
        # jacob = torch.autograd.functional.jacobian(exp_func, input_points)
        
        # ids = range(num_points)
        # ids1 = range(batch_size)
        # jacob = jacob[:, ids, :, :, ids, :][:, ids1, :, ids1, :]
        # for i in range(batch_size):
        #     for j in range(num_points):
        #         diff = torch.norm(jacob[i,j] - total_distortion[i,j])
        #         if diff > 1e-4:
        #             print(i,j,diff)
        # import pdb; pdb.set_trace()
        
        ret_dict.update({
            'pred_points': pred_points,
            'pred_shape': pred_points1, 
            # 'new_vertices': new_vertices.clone(),
            'target_points': target_points, 
            'temp_points': temp_points, 
        })
        
        self.forward_ret_dict = ret_dict
        return batch_dict

    def forward_two_directions(self, batch_dict, inverse=False):
        
        batch_size =  batch_dict['batch_size']
        temp_points = self.temp_vert.repeat(batch_size, 1,1)
        shape_points = batch_dict['shape_points']
    
        if not self.use_full:
            input_points = temp_points[:, self.fps_ids, :]
            target_points = shape_points[:, self.fps_ids, :]
        else:
            input_points = temp_points
            target_points = shape_points
        
        num_points = temp_points.shape[1]
        ret_dict = {
            'pose_param': batch_dict['pose_param'],
            'target_points': target_points,
            'batch_size': input_points.shape[0],
            'names': batch_dict['idx'],
            'input_points': input_points,
            'temp_points': temp_points, 
            'shape_points': shape_points, 
            'fps_ids': self.fps_ids, 
            'temp_face': self.temp_face, 
        }  

        total_distortion = None  
        pred_points = input_points.clone()
        pred_points1 = temp_points.clone() ###### 
        if self.predict_normal:
            normal_pred_total = batch_dict['normal_pred'].view(batch_size, self.num_layers, self.num_inner_layer, 3)
                
        for i in range(self.num_layers//2):
            var_pred = batch_dict['var_pred%d'%(i)]
            var_pred = var_pred.view(batch_size, -1, self.num_inner_layer)
    
            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i], pred_points.float().transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if self.predict_normal:
                normal_pred = normal_pred_total[:,i,:,:]
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, normal_pred)
            else:
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred)
            
            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                    pred_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if not self.use_full:   
                pred_points1, mid_dict1 = self.tutte_layers[i](pred_points1, var_pred)
            else:
                mid_dict1 = mid_dict
                pred_points1 = pred_points 
            
            if 'l%d_distortion'%(i) in mid_dict1.keys():
                distortion = mid_dict1['l%d_distortion'%(i)]
                if self.rotate_triplane:
                    distortion = torch.matmul(self.normal_matrices[i].transpose(1,2), distortion)
                    distortion = torch.matmul(distortion, self.normal_matrices[i], ) 
                    
                if total_distortion is None:
                    total_distortion = distortion 
                else: 
                    total_distortion = torch.matmul(distortion, total_distortion)
                batch_dict['distortion%d'%(i)] = distortion.clone()
                ret_dict.update({'distortion%d'%(i): distortion,}) 
                for j in range(self.num_inner_layer):
                    ret_dict.update({'distortion%d_%d'%(i, j): mid_dict1['l%d_distortion%d'%(i, j+1)],})
                    
            ret_dict.update({'var_pred%d'%(i): var_pred})
        
        if total_distortion is not None:
            ret_dict['total_distortion_1'] = total_distortion 
        
        ret_dict.update({
            'pred_points_1': pred_points,
            'pred_shape_1': pred_points1, 
            # 'new_vertices': new_vertices.clone(),
            'target_points': target_points, 
            'temp_points': temp_points, 
        })
        
        total_distortion = None  
        pred_points = target_points.clone()
        pred_points1 = shape_points.clone() 
        
        for i in range(self.num_layers//2, self.num_layers):
            var_pred = batch_dict['var_pred%d'%(i)]
            var_pred = var_pred.view(batch_size, -1, self.num_inner_layer)

            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i], pred_points.float().transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if self.predict_normal:
                normal_pred = normal_pred_total[:,i,:,:]
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, normal_pred)
            else:
                pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred)
            
            if self.rotate_triplane:
                pred_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                    pred_points.transpose(1,2)).transpose(1,2) # [B, N, 3]
            
            if not self.use_full:   
                pred_points1, mid_dict1 = self.tutte_layers[i](pred_points1, var_pred)
            else:
                mid_dict1 = mid_dict
                pred_points1 = pred_points 
            
            # batch_dict.update(mid_dict)
            if 'l%d_distortion'%(i) in mid_dict1.keys():
                distortion = mid_dict1['l%d_distortion'%(i)]
                if self.rotate_triplane:
                    distortion = torch.matmul(self.normal_matrices[i].transpose(1,2), distortion)
                    distortion = torch.matmul(distortion, self.normal_matrices[i], ) 
            
                if total_distortion is None:
                    total_distortion = distortion 
                else: 
                    total_distortion = torch.matmul(distortion, total_distortion)
                ret_dict.update({'distortion%d'%(i): distortion,}) 
                for j in range(self.num_inner_layer):
                    ret_dict.update({'distortion%d_%d'%(i, j): mid_dict1['l%d_distortion%d'%(i, j+1)],})
    
            ret_dict.update({'var_pred%d'%(i): var_pred})
        
        if total_distortion is not None:
            ret_dict['total_distortion'] = total_distortion #####################
        
        ret_dict.update({
            'pred_points_2': pred_points,
            'pred_shape_2': pred_points1,             
        })
        print(ret_dict['pred_points_1'].mean(), ret_dict['pred_points_1'].max(), ret_dict['pred_points_1'].min())
        if self.compute_inverse or not self.training: 
            pred_points = ret_dict['pred_points_1'] 
            # pred_points = ret_dict['pred_points_2'] 
            
            for i in range(self.num_layers-1, self.num_layers//2-1, -1):
                var_pred = batch_dict['var_pred%d'%(i)]
                var_pred = var_pred.view(batch_size, -1, self.num_inner_layer)

                if self.rotate_triplane:
                    pred_points = torch.matmul(self.normal_matrices[i], pred_points.float().transpose(1,2)).transpose(1,2) # [B, N, 3]

                if self.predict_normal:
                    normal_pred = normal_pred_total[:,i,:,:]
                    pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, normal_pred, inverse=True)
                else:
                    pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, inverse=True)

                if self.rotate_triplane:
                    pred_points = torch.matmul(self.normal_matrices[i].transpose(1,2), \
                        pred_points.transpose(1,2)).transpose(1,2) # [B, N, 3]

                if not self.use_full:   
                    pred_points1, mid_dict1 = self.tutte_layers[i](pred_points1, var_pred, inverse=True)
                else:
                    mid_dict1 = mid_dict
                    pred_points1 = pred_points 
            
            ret_dict.update({
                'pred_points': pred_points,
                'pred_shape': pred_points1, 
            })
        else:
            ret_dict.update({
                'pred_points': target_points,
                'pred_shape': shape_points, 
            })
            
            
        self.forward_ret_dict = ret_dict
        return batch_dict
      
    def get_evaluation_results(self):
        return self.forward_ret_dict
