import numpy as np
from scipy.spatial import Delaunay

import torch
import torch.nn as nn 

import torch_geometric 
from torch_sparse_solve import solve

class TutteLayer(nn.Module):
    def __init__(self, mesh, lap_dict, radius=1,):
        super(TutteLayer, self).__init__()
        self.vertices = mesh['vertices']
        self.edges = mesh['edges']
        self.faces = mesh['faces']
        self.radius = radius
        self.epsilon = 0.2
        self.epsilon_angle = 0.1
        self.circle_map = False
        
        self.use_sigmoid = True
        self.use_plain_scale = True
        self.w_scale = 100.0
        self.angle_scale = 100.0     
        self.angle_sigmoid = True 
        
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

    def forward(self, input_points, var_pred, inverse=False, depth=None):
        if inverse:
            return self.forward_inverse(input_points, var_pred)
 
        input_points_np = input_points.clone().detach().cpu().numpy()
        batch_size = input_points_np.shape[0]
        N_points = input_points_np.shape[1]
        input_points_np = np.reshape(input_points_np, (batch_size*N_points, 2))
        face_ids = Delaunay.find_simplex(self.tri, input_points_np.astype(np.float64), bruteforce=True)
        input_nodes = self.faces[face_ids]
        input_nodes = np.reshape(input_nodes, (batch_size, N_points, 3)).astype(np.int32)

        input_areas = self.get_areas(input_points, input_nodes) # [b, N, 3]
        new_vertices = self.tutte_embedding_sparse(var_pred)
        # import pdb; pdb.set_trace() 
        pred_points = self.get_tutte_from_triangle(new_vertices, input_nodes, input_areas)
        
        d_faces = np.reshape(self.faces[face_ids], (batch_size, N_points, 3)).astype(np.int32)
        distortions = self.compute_distortion(d_faces, self.vertices, new_vertices)
        return_dict = {'new_vertices': new_vertices, 'distortion': distortions}
        return pred_points, new_vertices, distortions 

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
        distortions = torch.zeros(batch_size* N_points, 2,2)
        
        return pred_points, new_vertices, distortions 

    def tutte_embedding_sparse(self, var_pred):
        """
        Args:
            W_var: [b, n_edge]
            angle_var: [b, n_bound]
        """
        W_var = var_pred[:, :self.num_edges]
        angle_var = var_pred[:, self.num_edges:self.num_edges+len(self.bound_verts)]
        
        angle_var = angle_var / self.angle_scale
        angle_var = torch.sigmoid(angle_var) * (1-2*self.epsilon_angle) + self.epsilon_angle
        
        
        if self.use_sigmoid:
            W_var = W_var/self.w_scale
            W_var = torch.sigmoid(W_var) * (1-2*self.epsilon) + self.epsilon
        
        n_vert = self.vertices.shape[0]
        batch_size = W_var.shape[0]
        
        if angle_var is None:
            angle_init = 2 * torch.pi * torch.arange(len(self.bound_verts)).float()/len(self.bound_verts)
            angle_init = angle_init.unsqueeze(0).repeat(batch_size, 1)
            
        else:
            new_angle = angle_var/angle_var.sum(1).unsqueeze(1)
            angle_init = torch.cumsum(new_angle, dim=1) * 2 * torch.pi # [b, n_bound]

        angle_init = angle_init.view(-1, )
        bound_pos = torch.zeros((batch_size * len(self.bound_verts), 2)) # [b, n_bound, 2]

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

        A = torch.sparse_coo_tensor(lap_indices.cpu(), lap_values.double().cpu(), (batch_size, len(self.interior_verts), len(self.interior_verts)))
        b = b[:, self.interior_verts].double()
        x =  solve(A, b.cpu())

        out_pos = torch.zeros(batch_size, n_vert, 2)
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

class TutteVaryingNormal(nn.Module):
    def __init__(self, mesh, lap_dict, normal_matrices,radius=1, prefix='',):
        super(TutteVaryingNormal, self).__init__()
        
        self.normal_matrices = normal_matrices
        self.num_layer = self.normal_matrices.shape[0]
        
        self.tutte_layers = nn.ModuleList([TutteLayer(mesh, lap_dict, radius) \
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
            pred_points, new_vertices, distortions = self.tutte_layers[i](cur_points,  var_pred[:,:,i], depth=rot_points[:,:,2])
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

class TutteModelLearning(nn.Module):
    def __init__(self, mesh, bound_verts, normals, num_layer=1, radius=1):
        super(TutteModelLearning, self).__init__()
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
#         self.normals = np.array([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
#                                  [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]], np.float32)
        self.normals = normals 
        # self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        # print("normals", self.normals)

        self.normal_matrices = np.zeros((self.normals.shape[0], 3,3))
        for i in range(self.normals.shape[0]):
            self.normal_matrices[i] = self.rotation_matrix_from_vectors(self.normals[i])
        self.normal_matrices = torch.from_numpy(self.normal_matrices).float().unsqueeze(1)
        
        if 0:
            self.tutte_layers = nn.ModuleList([TuttePredictingNormal(mesh, self.lap_dict, prefix='l%d_'%(i) ) \
                for i in range(self.num_layers)]) 
        else:
            self.tutte_layers = nn.ModuleList([TutteVaryingNormal(mesh, self.lap_dict,self.normal_matrices, prefix='l%d_'%(i) ) \
                for i in range(self.num_layers)]) 
            
    def forward(self, input_points, var_pred, inverse=False):
        if inverse:
            return self.forward_inverse(input_points, var_pred)
        
        pred_points = input_points.clone() 
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layers):
            pred_points, mid_dict = self.tutte_layers[i](pred_points, var_pred, inverse=inverse) 
            
            # distortions = mid_dict['L%d_distortion'%(i)]
            # if total_distortion is None:
            #     total_distortion = distortions 
            # else: 
            #     total_distortion = torch.matmul(distortions, total_distortion)
            # return_dict.update(mid_dict)
        return_dict['pred_points'] = pred_points
        # return_dict['total_distortion'] = total_distortion
        # return_dict['new_vertices'] = new_vertices 
        return return_dict
    
    def forward_inverse(self, input_points, var_pred):
        pred_points = input_points.clone() 
        return_dict = {} 
        total_distortion = None 
        for i in range(self.num_layers):
            pred_points, mid_dict = self.tutte_layers[self.num_layers-i-1](pred_points, var_pred, inverse=True) 
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