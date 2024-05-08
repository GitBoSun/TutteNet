import time
import os 
import trimesh
import numpy as np
import matplotlib 

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import NearestNeighbors

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
    for i in range(N):
        c = cmap(dis[i])
        c = [int(x*255) for x in c]
        
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2], \
                                            c[0], c[1], c[2] )) 

    fout.close()

def filter_outliers(mesh=None, path=None,remove_small=False, thres=0.02, small_thres=100):
    if mesh is None:
        mesh = trimesh.load(path)

    masked_points = mesh.vertices  
    num_points= masked_points.shape[0]

    tree = NN(n_neighbors=10).fit(masked_points)
    dists, indices = tree.kneighbors(masked_points)
    e0 = np.arange(num_points).repeat(10)
    e1 = indices.reshape(-1)
    mask = dists.reshape(-1) < thres
    e0, e1 = e0[mask], e1[mask]

    graph = csr_matrix((np.ones_like(e0), (e0, e1)), shape=(num_points, num_points))
    n_components, labels = connected_components(graph, directed=False)
    
    if not remove_small:
        max_num = 0 
        for i in range(n_components):
            if (labels==i).sum()>max_num:
                idx = i 
                max_num = (labels==i).sum() 
        print(idx, max_num)
        new_points = masked_points[labels==idx]
        new_mesh = trimesh.Trimesh(vertices=new_points,faces=[])
    else:
        mask = labels>=-1 
        for i in range(n_components):
            if (labels==i).sum()<small_thres:
                # print(i)
                mask = mask & (labels!=i) 
                 
        print(mask.sum())
        new_points = masked_points[mask]
        new_mesh = trimesh.Trimesh(vertices=new_points,faces=[])
    return new_mesh 

def get_dense_mesh(density_path):
    density_mesh = trimesh.load(os.path.join(density_path,'0_5.ply'))
    density_points = density_mesh.vertices 
    for i in range(1,40):
        try:
            density_mesh_tmp = trimesh.load(os.path.join(density_path,'%d_5.ply'%(i)))
        except:
            continue 
        print(i)
        if density_points.shape[0]>2000000:
            break 
        density_points = np.concatenate((density_points, density_mesh_tmp.vertices), axis=0)

    dense_mesh = trimesh.Trimesh(vertices=density_points,faces=[])
    dense_mesh = filter_outliers(dense_mesh, thres=0.05) 
    dense_mesh.export(os.path.join(density_path,'dense_mesh.ply'))
    return dense_mesh


def get_free_points(point_mesh, body_mesh, handle_mesh):

    # get rest of the points as moving part
    tree = NN(n_neighbors=10,radius=0.05).fit(point_mesh.vertices )
    dists, indices = tree.kneighbors(np.concatenate((body_mesh.vertices, handle_mesh.vertices), axis=0))

    moving_mask = np.ones(point_mesh.vertices.shape[0]).astype(np.bool)
    moving_mask[indices.reshape(-1)] = False 
    moving_points = point_mesh.vertices[moving_mask]
    # print('begin', moving_points.shape)

    nbrs = tree.radius_neighbors(moving_points, 0.05, return_distance=False)
    enlarged_mask = np.unique(np.concatenate(nbrs))
    moving_points = point_mesh.vertices[enlarged_mask]
    # print('enlarged', moving_points.shape)
    return moving_points 

def set_handle_target(handle_mesh, angles, trans_vec):
    # set handle target 
    num1 = handle_mesh.vertices.shape[0]
    total_handle_points = np.zeros((num1, 3))

    handle_mean = handle_mesh.vertices.mean(0)
    centered_handle = handle_mesh.vertices - np.expand_dims(handle_mean, 0) 

    alpha = angles[0] 
    rot_matrix1 = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
    centered_handle = np.matmul(centered_handle,rot_matrix1 )
    alpha1 = angles[1]
    rot_matrix2 = np.array([[np.cos(alpha1),0, np.sin(alpha1)], [0,1,0], [-np.sin(alpha1),0, np.cos(alpha1)]])
    centered_handle = np.matmul(centered_handle,rot_matrix2 )
    alpha2 = angles[2]
    rot_matrix3 = np.array([[np.cos(alpha2), np.sin(alpha2), 0], [-np.sin(alpha2), np.cos(alpha2), 0],[0,0,1]])
    centered_handle = np.matmul(centered_handle,rot_matrix3)

    translated_handle = centered_handle + np.expand_dims(handle_mean, 0)  
    # trans_vec = np.array([[0., 0.45, 0.1]]) # tmp3 

    translated_handle = translated_handle + trans_vec
    total_handle_points[:num1] = translated_handle 
    translated_handle = total_handle_points 

    total_rotation_matrix = np.matmul(rot_matrix1, rot_matrix2)
    total_rotation_matrix = np.matmul(total_rotation_matrix, rot_matrix3)
    return translated_handle, total_rotation_matrix 




