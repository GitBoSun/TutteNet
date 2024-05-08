import os
import numpy as np
import torch
import pickle
COLORS = np.array([
        [0.3, 0.3, 0.3], # 0
        #[1,0,0],
        [1.0,140.0/255.0,0],
        [30.0/255.0,144.0/255.0, 1.0],
        #[1,0,0],
        #[0.6, 0.1, 0.8], # 3
        [50.0/255.0,205.0/255.0,50.0/255.0],
        [1.0,0,1.0], #[0.2, 0.1, 0.9],
        [127/255.0,255/255.0,0],
        [255.0/255.0, 51.0/255.0, 51.0/255.0 ],  #[0,1,0], # 6
        [1.0,0.0,0.0], #[1.0,1.0,0],  # [0.8,0.8,0.8],
        [0.0, 0.8, 0.8],
        [210/255.0,105/255.0,30/255.0], #[0.05, 0.05, 0.3],
        [0.8, 0.6, 0.2], # 10 
        [148/255.0,0,211/255.0],
        [127/255.0,255/255.0,212/255.0], # 12
        [255/255.0,215/255.0,0], #[0.2, 0.5, 0.8],
        [0.0, 128.0/255.0, 0],
        [154/255.0,205/255.0,50/255.0],
        [230/255.0,230/255.0,250/255.0], # 16
        [240/255.0,230/255.0,140/255.0],
        [176/255.0,224/255.0,230/255.0], #[0.8, 0.2, 0.8],
        [255/255.0,99/255.0,71/255.0], # 19
        [0,1,1], #[0., 1, 0.3],
        [100/255.0,149/255.0,237/255.0],
        [255/255.0,105/255.0,180/255.0]
        ]).astype(np.float32)


def get_box_points(center, dimensions, yaw):
        x, y, z = dimensions
        points = np.array([[-x/2, -y/2, -z/2], [-x/2, -y/2, z/2],
                           [-x/2, y/2, -z/2], [-x/2, y/2, z/2],
                           [x/2, -y/2, -z/2], [x/2, -y/2, z/2],
                           [x/2, y/2, -z/2], [x/2, y/2, z/2],])
        R = np.array([
            [np.cos(-yaw), -np.sin(-yaw), 0],
            [np.sin(-yaw), np.cos(-yaw), 0],
            [0,0,1]])
        points = np.matmul(points, R)
        points = points + center.reshape(-1, 3)
        return points


def transform(lidar_data, M):
    def to_homogeneous(xyz):
        return np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    def from_homogeneous(xyz):
        return xyz[:, 0:3]

    lidar_data_new = lidar_data.copy()

    xyz = to_homogeneous(lidar_data_new[:, 0:3])
    xyz = np.matmul(M, xyz.T).T
    xyz = from_homogeneous(xyz)

    lidar_data_new[:, 0:3] = xyz

    return lidar_data_new


def write_ply_color(points, labels,  out_filename, label_color=False):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    N = points.shape[0]
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
        #c = pyplot.cm.hsv(labels[i])
        #c = colors[i,:]
        #c = [int(x*255) for x in c]
        if not label_color:
            fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2], \
                   COLORS[labels[i], 0]*255, COLORS[labels[i], 1]*255, COLORS[labels[i], 2]*255))
        else:
            fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2], \
                labels[i,0],  labels[i,1], labels[i,1]))

    fout.close()
