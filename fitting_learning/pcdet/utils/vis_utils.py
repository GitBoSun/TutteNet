import os
import cv2
import glob
import trimesh
import numpy as np
import torch
from psbody.mesh import Mesh 
import matplotlib.pyplot as plt
#import colormap
import matplotlib 
from matplotlib import cm


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
    #normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05)

    #colormap possible values = viridis, jet, spectral
    # cmap = plt.cm.jet
    # cmap.set_over(0.02)

    for i in range(N):
        # colors = plt.cm.hsv(labels[i])
        # colors = cmap(labels[i])
        # colors = cm.jet(norm(labels[i]),bytes=True) 
        colors = cm.seismic(norm(labels[i]),bytes=True) 

        # c = colors[i,:]
        # c = [int(x*255) for x in c]
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2], \
                colors[0],  colors[1], colors[1]))

    fout.close()
    
def save_3d_results(pred_dicts, out_path, epoch=-1, val=False, vis_prefix=""):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    names = pred_dicts['names'].cpu().numpy()
     
    temp_shape = pred_dicts['temp_points'].cpu().numpy()
    shape_points = pred_dicts['shape_points'].cpu().numpy()
    # pred_shape = pred_dicts['pred_shape'].detach().cpu().numpy()
    pred_shape = pred_dicts['pred_points'].detach().cpu().numpy()
    temp_face = pred_dicts['temp_face']
    # source_face = temp_face 
    source_face = pred_dicts['source_face'].cpu().numpy()

    input_points = pred_dicts['input_points'].cpu().numpy()
    target_points = pred_dicts['target_points'].cpu().numpy()
    pred_points = pred_dicts['pred_points'].detach().cpu().numpy() 
    j_loss_i = pred_dicts['j_loss_i'].detach().cpu().numpy() 
    
    batch_size = pred_points.shape[0]
    # batch_size = min(batch_size, 16)
    
    if 'normal_pred' in pred_dicts.keys():
        pred_normal = pred_dicts['normal_pred'].detach().cpu().numpy()  
    else:
        pred_normal = None 

    if 'total_points' in pred_dicts.keys():
        total_points = pred_dicts['total_points'].detach().cpu().numpy()  
        total_points_rot = pred_dicts['total_points_rot'].detach().cpu().numpy()  
    else:
        total_points = None 
        
    if 'var_pred0' in pred_dicts.keys():
        total_var_pred = pred_dicts['var_pred0'].detach().cpu().numpy()
    else:
        total_var_pred = None 
    # prefix = 'amass_'
    # alpha = -np.pi/2 
    
    alpha = 0
    rot_matrix = np.array([[1,0,0], [0,np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)],])
    for bi in range(batch_size):
        files = glob.glob(os.path.join(out_path, "*_%d*.ply"%(names[bi])))
        for f in files:
            if not val:
                os.system("rm %s"%(f))
        # print('hh', pred_shape[bi].max(0), pred_shape[bi].min(0))
        pred_points_i = np.matmul(pred_shape[bi], rot_matrix)
        mesh3 = Mesh(v=pred_points_i, f=source_face)        
        mesh3.write_ply(os.path.join(out_path, '%sep%d_%d_pred.ply'%(vis_prefix, int(epoch), int(names[bi]))))
        mesh3 = Mesh(v=np.matmul(shape_points[bi,],rot_matrix ), f=temp_face)
        mesh3.write_ply(os.path.join(out_path, '%sep%d_%d_gt.ply'%(vis_prefix, int(epoch), int(names[bi]))))
        
        if pred_points_i.shape[0]==shape_points[bi,].shape[0]:
            errors = j_loss_i[bi]
            # errors = np.sqrt(np.square(pred_points_i - shape_points[bi,]).sum(1))
            # write_ply_color(pred_points_i, errors, os.path.join(out_path, '%sep%d_%d_error.ply'%(vis_prefix, int(epoch), int(names[bi]))), )
            # np.save(os.path.join(out_path, '%sep%d_%d_jloss.npy'%(vis_prefix, int(epoch), int(names[bi]))), errors)
        mesh3 = Mesh(v=np.matmul(temp_shape[bi,],rot_matrix ), f=temp_face)
        mesh3.write_ply(os.path.join(out_path, '%sep%d_%d_temp.ply'%(vis_prefix, int(epoch), int(names[bi]))))
        
        # if total_points is not None:
        #     print('hh')
        #     for ii in range(24):
        #         mesh3 = Mesh(v=np.matmul(total_points[ii,bi],rot_matrix ), f=source_face)
        #         mesh3.write_ply(os.path.join(out_path, '%sep%d_%d_layer%d.ply'%(vis_prefix, int(epoch), int(names[bi]), ii)))
        #         mesh3 = Mesh(v=np.matmul(total_points_rot[ii,bi],rot_matrix ), f=source_face)
        #         mesh3.write_ply(os.path.join(out_path, '%sep%d_%d_layer%d_rot.ply'%(vis_prefix, int(epoch), int(names[bi]), ii)))
                
        
        
        # if pred_normal is not None:
        #     # print("$$$$", pred_normal[bi])
        #     n_normal = pred_normal[bi].shape[0]
        #     normal_points = np.zeros((n_normal, 100, 3))
        #     normal_colors = np.zeros((n_normal, 100,))
        #     for i in range(n_normal): 
        #         for j in range(100):
        #             normal_points[i, j] = pred_normal[bi][i]*float(j/100) 
        #         normal_colors[i] = float(i/n_normal)
        #     write_ply_color(normal_points.reshape((n_normal*100, 3)), normal_colors.reshape(n_normal*100), os.path.join(out_path, '%sep%d_%d_normal.ply'%(vis_prefix, int(epoch), int(names[bi]))), )
        
        
        
        # save var_pred and normal  ##### 
        if 'var_pred0' in pred_dicts.keys():
            np.savez(os.path.join(out_path, '%sep%d_%d_var_pred.npz'%(vis_prefix, int(epoch), int(names[bi]))), \
                var_pred=total_var_pred[bi], pred_normal=pred_normal[bi])
        
            
        # mesh3 = Mesh(v=np.matmul(input_points[bi,],rot_matrix ), f=[])
        # mesh3.write_ply(os.path.join(out_path, 'ep%d_%d_handle_input.ply'%(epoch, names[bi])))
        # mesh3 = Mesh(v=np.matmul(target_points[bi,],rot_matrix ), f=[])
        # mesh3.write_ply(os.path.join(out_path, 'ep%d_%d_handle_gt.ply'%(epoch, names[bi])))
        # mesh3 = Mesh(v=np.matmul(pred_points[bi,],rot_matrix ), f=[])
        # mesh3.write_ply(os.path.join(out_path, 'ep%d_%d_handle_pred.ply'%(epoch, names[bi])))
        

def plot_embed_results(pred_dicts, out_path, epoch=-1, val=False, vis_prefix=""):
    if 'temp_points' in pred_dicts.keys():
        save_3d_results(pred_dicts, out_path, epoch, val, vis_prefix)
        return 
    
    # print('saving results...')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    size1 = 2
    size2 = 4
    fsize = 8
    loss = torch.square(pred_dicts['pred_points'] - pred_dicts['target_points']).mean(2).mean(1).detach().cpu().numpy() 
    e_param = pred_dicts['e_param']
    angles = e_param[:, 2].cpu().numpy()  
    
    if pred_dicts['pred_points'].shape[2]==3:
        names = pred_dicts['names'].cpu().numpy()
        pred_points = pred_dicts['pred_points'].detach().cpu().numpy()
        target_points = pred_dicts['target_points'].cpu().numpy()
        input_points = pred_dicts['input_points'].cpu().numpy()
        batch_size = pred_points.shape[0]
        batch_size = min(batch_size, 16)
        for bi in range(batch_size):
            name = names[bi]
            files = glob.glob(os.path.join(out_path, "*_%s*.png"%(name)))
            for f in files:
                if not val:
                    os.system("rm %s"%(f))
            fig = plt.figure(figsize=(fsize, fsize))
               
            colors = input_points[bi].sum(1)
            colors = colors - colors.min()
            colors = colors / colors.max() 
            
            plt.subplot(3, 3, 1)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(input_points[bi, :,0],input_points[bi, :,1], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 2)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(input_points[bi, :,0],input_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 3)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(input_points[bi, :,1],input_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 4)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(target_points[bi, :,0],target_points[bi, :,1], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 5)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(target_points[bi, :,0],target_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 6)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(target_points[bi, :,1],target_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 7)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(pred_points[bi, :,0],pred_points[bi, :,1], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 8)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(pred_points[bi, :,0],pred_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.subplot(3, 3, 9)
            plt.axis('equal')
            plt.xlim(-1.2, 1.2) 
            plt.ylim(-1.2, 1.2)
            plt.scatter(pred_points[bi, :,1],pred_points[bi, :,2], c=colors, cmap='rainbow', s=1)
            plt.savefig(os.path.join(out_path, "%d_%s_l%.2f_pred.png"%(epoch, name,  loss[bi]*100)),)
        return  

    if 'new_vertices' not in pred_dicts.keys():
        names = pred_dicts['names'].cpu().numpy()
        pred_points = pred_dicts['pred_points'].detach().cpu().numpy()
        target_points = pred_dicts['target_points'].cpu().numpy()
        batch_size = pred_points.shape[0]
        batch_size = min(batch_size, 16)
        for bi in range(batch_size):
            name = names[bi]
            files = glob.glob(os.path.join(out_path, "*_%s*.png"%(name)))
            for f in files:
                if not val:
                    os.system("rm %s"%(f))

            fig = plt.figure(figsize=(fsize, fsize))
            colors = (np.arange(target_points.shape[1]).astype(np.float32) % target_points.shape[1])/target_points.shape[1]
            plt.scatter(target_points[bi, :,0],target_points[bi, :,1], c=colors, cmap='Blues', s=size1 )
            
            colors = (np.arange(target_points.shape[1]).astype(np.float32) % target_points.shape[1])/target_points.shape[1]
            plt.scatter(pred_points[bi,:,0],pred_points[bi,:,1], c=colors, cmap='rainbow', s=size2)
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.axis('equal')
            plt.savefig(os.path.join(out_path, "%d_%s_a%.2f_l%.2f_pred.png"%(epoch, name, angles[bi], loss[bi]*100)),)
        return

    names = pred_dicts['names'].cpu().numpy()
    pred_points = pred_dicts['pred_points'].detach().cpu().numpy()
    new_vertices = pred_dicts['new_vertices'].detach().cpu().numpy()
    target_points = pred_dicts['target_points'].cpu().numpy()
    simplices = pred_dicts['simplices']
    vertices = pred_dicts['vertices'].cpu().numpy()

    input_points = pred_dicts['input_points'].detach().cpu().numpy()
    batch_size = pred_points.shape[0]
    batch_size = min(batch_size, 16)

    for bi in range(batch_size):
        name = names[bi]

        files = glob.glob(os.path.join(out_path, "*_%s*.png"%(name)))
        for f in files:
            if not val:
                os.system("rm %s"%(f))

        fig = plt.figure(figsize=(fsize, fsize))
        plt.triplot(vertices[:,0], vertices[:,1], simplices, c='lightsteelblue',) # lightslategrey
        plt.scatter(target_points[bi, :,0],target_points[bi, :,1], c='b', s=size1 )
        colors = (np.arange(input_points.shape[1]).astype(np.float32) % input_points.shape[1])/input_points.shape[1]
        plt.scatter(input_points[bi,:,0],input_points[bi,:,1], c=colors, cmap='rainbow', s=size2)
        plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
        plt.axis('equal')
        plt.savefig(os.path.join(out_path,  "%d_%s_a%.2f_l%.2f_or.png"%(epoch, name, angles[bi], loss[bi]*100)),)
        # plt.show()

        for i in range(20):
            
            if 'new_vertices%d'%(i+1) in pred_dicts.keys() and i%1==0:
                # print("layer", i)

                new_vertices1 = pred_dicts['new_vertices%d'%(i+1)][bi]
                pred_points1 = pred_dicts['pred_points%d'%(i+1)][bi]
                pos_np = new_vertices1.detach().cpu().numpy()

                fig = plt.figure(figsize=(fsize, fsize))
                plt.triplot(pos_np[:,0], pos_np[:,1], simplices, c='lightsteelblue')
                colors = (np.arange(target_points.shape[1]).astype(np.float32) % target_points.shape[1])/target_points.shape[1]
                plt.scatter(target_points[bi,:,0],target_points[bi,:,1], c=colors, cmap='Blues', s=size1 )
                plt.scatter(pred_points1.detach().cpu().numpy()[:,0],pred_points1.detach().cpu().numpy()[:,1],  c=colors, cmap='rainbow', s=size2)

                plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
                plt.axis('equal')
                plt.savefig(os.path.join(out_path, "%d_%s_middle_%d.png"%(epoch, name, i)),)
                # plt.show()

                # fig = plt.figure(figsize=(10, 10))
                # plt.triplot(vertices[:,0], vertices[:,1], simplices, c='lightsteelblue')

                # plt.scatter(target_points.numpy()[bi,:,0],target_points.numpy()[bi,:,1], c='b', s=size1 )
                # colors = (3*np.arange(input_points.shape[1]).astype(np.float32) % input_points.shape[1])/input_points.shape[1]
                # plt.scatter(pred_points1.detach().numpy()[:,0],pred_points1.detach().numpy()[:,1],  c=colors, cmap='rainbow', s=size2)

                # plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
                # plt.axis('equal')
                # plt.show()

        fig = plt.figure(figsize=(fsize, fsize))
        plt.triplot(new_vertices[bi,:,0], new_vertices[bi,:,1], simplices, c='lightsteelblue')
        colors = (np.arange(input_points.shape[1]).astype(np.float32) % input_points.shape[1])/input_points.shape[1]
        plt.scatter(target_points[bi,:,0],target_points[bi,:,1], c=colors, cmap='Blues', s=size1)
        plt.scatter(pred_points[bi,:,0],pred_points[bi,:,1],  c=colors, cmap='rainbow', s=size2)

        plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
        plt.axis('equal')
        plt.savefig(os.path.join(out_path, "%d_%s_a%.2f_l%.2f_out.png"%(epoch, name, angles[bi], loss[bi]*100)),)
        # plt.show()
        # print('saved in', out_path)


