import _init_path
import argparse
import datetime
import glob
import os
import time 
import numpy as np 
import trimesh
from typing import Tuple
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt
from eval_utils.eval_utils import eval_one_epoch

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import nvdiffrast.torch as dr
import imageio
from PIL import Image 
from psbody.mesh import Mesh 
    
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, freeze_modules_except
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from pcdet.models import load_data_to_gpu

import clip
import sys 
sys.path.append("/home/bos/projects/dinov2")
from dinov2.data.transforms import make_classification_eval_transform 

# os.environ['CUDA_VISIBLE_DEVICES']="4,5,6,7"


def _warmup(glctx):
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device="cuda", **kwargs)

    pos = tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
        dtype=torch.float32,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])


def vertex_color_render(
    vertices: torch.Tensor,  # B,V,3,
    faces: torch.Tensor,  # V,3,
    vertices_color: torch.Tensor,  # V,3,
    uv: torch.Tensor,  # V,3,
    mv: torch.Tensor,  # B,4,4
    proj: torch.Tensor ,  # B,4,4
    image_size: Tuple[int, int],
    texture: torch.Tensor = None,  # B,H,W,3
    glctx: dr.RasterizeCudaContext = None,
) -> torch.Tensor:  # B,H,W,4
    # Number of vertices
    V = vertices.shape[1]
    B = vertices.shape[0]

    # put everything on the GPU
    device = vertices.device
    faces = faces.to("cuda:0").type(torch.int32)

    vertices = vertices.to("cuda:0").float()
    # uv = uv.to("cuda:0").float()
    if vertices_color is not None:
        colors = vertices_color.to("cuda:0").float()

    # mv = mv.to("cuda:0").float()
    # proj = proj.to("cuda:0").float()

    if texture is not None:
        texture = texture.to("cuda:0").float().contiguous()

    # Change the type of faces to int32
    faces = faces.type(torch.int32)

    # Add a homogeneous coordinate to the vertices
    vert_hom = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4c

    # Transform the vertices to clip space
    # vertices_clip = vert_hom @ mv.transpose(-2, -1) @ proj.transpose(-2, -1)  # C,V,4
    vertices_clip = vert_hom @ mv.transpose(-2, -1) 
    
    # orthographic
    # Change of convention to OPENGL CLIP Space. Z is pointing from the viewer into the screen
    vertices_clip[:, :, 2] = -vertices_clip[:, :, 2]
    # Memory convention in the rendered image. Either
    vertices_clip[:, :, 1] = -vertices_clip[:, :, 1]
    
    vertices_clip[:, :, 2] += 0.2
    # or
    # col = torch.flip(col, 1)

    # Check if gltctx is provided, otherwise create a new one
    if glctx is None:
        # glctx = dr.RasterizeGLContext()
        glctx = dr.RasterizeCudaContext(torch.device("cuda"))
        _warmup(glctx)
    
    # Rasterize data
    # print(vertices_clip.shape, faces.shape)
    rast_out, rast_out_db = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=True)  # C,H,W,4

    if texture is None:
        # interpolate depth for debugging
        # col, _ = dr.interpolate(vertices_clip[0,:,2:3].contiguous().repeat(1,3).contiguous(), rast_out, faces)  # C,H,W,3
        col, _ = dr.interpolate(colors, rast_out, faces)  # C,H,W,3
        # create alpha channel
    else:
        texc, texd = dr.interpolate(uv.float(), rast_out, faces, rast_db=rast_out_db, diff_attrs="all")
        col = dr.texture(texture, texc, texd, filter_mode="linear-mipmap-linear", max_mip_level=9)
    alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
    depth = rast_out[:, :, :, 2]  # C,H,W,1
    # if debugging with depth
    #  (col+1)/2*alpha + col*0*(1-alpha)

    # Add alpha channel
    col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
    # Anti-aliasing
    col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
    # col = col - col.min() / (col.max() - col.min())

    return col, depth  # C,H,W,4

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('data_cfg_file', type=str, default=None, help='specify the data config for training')
    parser.add_argument('opt_cfg_file', type=str, default=None, help='specify the optimizer config for training')
    parser.add_argument('--vis_cfg_file', type=str, default=None, help='specify the visualizer config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--scale', type=float, default=None, required=False, help='the scale of model')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    # parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--local-rank',dest='local_rank' ,type=int, default=0, help='local rank for distributed training')

    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, help='')
    parser.add_argument('--eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--ema', action='store_true', default=False, help='')
    parser.add_argument('--refinement', action='store_true', default=False, help='')
    parser.add_argument('--eval_interval',  type=int, default=10, help='eval_interval')
    parser.add_argument('--train_interval',  type=int, default=10, help='train_interval')
    parser.add_argument('--vis_prefix',  type=str, default="", help='train_interval')
    parser.add_argument('--save_train',  type=bool, default=True, help='train_interval')
    parser.add_argument('--start_idx',  type=int, default=0, help='eval_interval')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_yaml_file(args.data_cfg_file, cfg.DATA_CONFIG)
    if args.vis_cfg_file is not None:
        cfg_from_yaml_file(args.vis_cfg_file, cfg.MODEL)
    if args.opt_cfg_file is not None:
        cfg_from_yaml_file(args.opt_cfg_file, cfg)
    print(cfg)
    dataset_tag = args.data_cfg_file.split('dataset_configs/')[-1].replace('/', '_')
    cfg.TAG = Path(args.cfg_file).stem + '/' + Path(dataset_tag).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if args.scale is not None:
        logger.info(f'Setting model scale to {args.scale}, overwriting scale in config file.')
        cfg.MODEL['SCALE'] = args.scale

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))


    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )
    
    total_it_each_epoch = 1000000//args.batch_size 
    out_path = "rendered_data_simple_1M"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    out_path_im = "rendered_images_simple_1M"
    if not os.path.exists(out_path_im):
        os.makedirs(out_path_im)
    
    device  = "cuda"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_preprocess = make_classification_eval_transform()
    dataloader_iter = iter(train_loader)
    
    start_idx = args.start_idx 
    for cur_it, batch_dict in enumerate(train_loader):
        # if os.path.exists(os.path.join(out_path, "%d_%d.npz"%(start_idx, cur_it))):
        #     print('exists', start_idx, cur_it)
        #     continue 
        t = time.time() 
        load_data_to_gpu(batch_dict)
    
        # pos_or = batch_dict["temp_points"].float()
        pos_or = batch_dict["shape_points"].float()

        faces = train_set.temp_face 
        tri = torch.from_numpy(faces.astype(np.int32)).int().cuda()
        shape_param = batch_dict["shape_param"].float()
        pose_param = batch_dict["pose_param"].float()
        batch_size = pos_or.shape[0]
        col = torch.ones(( pos_or.shape[1], 3), dtype=torch.float32).cuda()
        
        # mv = torch.eye(4).cuda().unsqueeze(0).float().repeat(batch_size, 1,1) 
        # mv[1, 3] += 0.3
        mv = torch.eye(4).cuda().float()
        # mv[1, 3] += 0.3 
        mv = mv.unsqueeze(0).repeat(batch_size, 1,1) 
        
        clip_features = torch.zeros(batch_size, 8, 512).cuda()  
        dino_features = torch.zeros(batch_size, 8, 384).cuda()   
        
        for angle_i in range(8):
            alpha_y = torch.tensor(2*torch.pi * angle_i /8)
            rot_y = torch.tensor([[torch.cos(alpha_y), 0, torch.sin(alpha_y)], [0, 1, 0], [-torch.sin(alpha_y), 0, torch.cos(alpha_y)]])
            rot_y = rot_y.unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
            pos = torch.matmul(pos_or, rot_y)
            with torch.no_grad():
                color, depth = vertex_color_render(pos, tri, col, uv=None, mv=mv, proj=None, image_size=(256, 256))
            
            clip_images = torch.zeros(batch_size, 3, 224, 224).cuda()  
            dino_images = torch.zeros(batch_size, 3, 224, 224).cuda() 
            
            for j in range(batch_size):
                img_or = depth.cpu().numpy()[j] 
                img = img_or + 1
                img = img / 2
                img = 1 - img 
                img = img * (1-(img_or==0).astype(np.float32)) 
                img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
                tmp_path = os.path.join(out_path_im, '%d_%d.png'%(start_idx, j,)) 
                imageio.imsave(tmp_path, img)

                image = clip_preprocess(Image.open(tmp_path)).unsqueeze(0).to(device)
                clip_images[j] = image 
                image_dino = dinov2_preprocess(Image.open(tmp_path).convert("RGB")).unsqueeze(0).to(device) 
                dino_images[j] = image_dino 
                

            with torch.no_grad():
                clip_features[:, angle_i] = clip_model.encode_image(clip_images)
                dino_features[:, angle_i] = dinov2_vits14(dino_images)
        # print(shape_param.shape)
        # print(clip_features.shape, dino_features.shape)            
        np.savez(os.path.join(out_path, "%d_%d.npz"%(start_idx, cur_it)), shape_param=shape_param.cpu().numpy(), pose_param=pose_param.cpu().numpy(),
                     clip_features =clip_features.cpu().numpy(),  dino_features=dino_features.cpu().numpy(),)
        print('saved',start_idx, cur_it, time.time()-t )

def render_one_mesh():
    device  = "cuda"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_preprocess = make_classification_eval_transform()

    # name = 'manikin_1100'
    # name = 'manikin_1100_ground'
    # name = 'girl_1100_1.2'
    # name = 'siming3'
    name = 'bo3'
    out_path = 'single_rendered/%s'%(name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path_im = 'single_rendered/%s_ims'%(name)
    if not os.path.exists(out_path_im):
        os.makedirs(out_path_im)
        
    temp_path = '/home/bosun/projects/PCPerception/tools/template/smpl_template.obj'
    scale_mesh = Mesh(filename=temp_path)
    scale_factor = 1.4 
    
    vert1 = scale_mesh.v 
    print(vert1.min(0), vert1.max(0))

    min1 = vert1.min(0)
    vert1 = vert1 - min1
    max1_1 = vert1.max(0)
    min1_1 = vert1.min(0)
    vert1 = vert1 / (max1_1[1] - min1_1[1])
    min1_2 = vert1.min(0) 
    max1_2 = vert1.max(0) 
    vert1 = (vert1 - (max1_2 -min1_2)/2)*scale_factor 
     
    # mesh_path = '/home/bos/projects/CAPE/dataset/cape_release/meshes/00032/shortlong_hips/canonical/shortlong_hips.000010.obj'
    # mesh_path = '/home/bos/projects/skeleton-builder/big-buck-bunny.obj'
    # mesh_path = '/home/bos/projects/CAPE/results/CAPE-affineconv_nz64_pose32_clotype32_male/sample_vary_clotype/clotype_longlong_0000.obj'
    # mesh_path = './nerf_meshes/manikin_rot_scale_simp_remove_rot.ply'
    # mesh_path = './nerf_meshes/1100_pred_shape_remove_rot.ply'
    # mesh_path = './nerf_meshes/1100_pred_shape_rot.ply'
    # mesh_path = './nerf_meshes/girl/1100_pred_shape_rot_1.2.ply'
    # mesh_path = './nerf_meshes/robo/20000_rot_scale_simp_rot.ply'
    # mesh_path = './nerf_meshes/180000_pcd_nofloor_recons_rot.ply'
    mesh_path = './nerf_meshes/siming3/marching_cube_2_rot.ply'
    mesh_path = '/home/bosun/projects/nerf/positions/bo_nerfacto/dense_mesh_5_nofloor_marching_cube.ply'



    # mesh = Mesh(filename=mesh_path)
    # vert2 = mesh.v 
    mesh = trimesh.load(mesh_path)
    vert2 = mesh.vertices 
    print(vert2.min(0), vert2.max(0))
    # import pdb; pdb.set_trace() 
    
    vert2 = vert2 - min1
    vert2 = vert2 / (max1_1[1] - min1_1[1])
    vert2 = (vert2 - (max1_2 - min1_2)/2)*scale_factor 
    
    print(vert2.min(0), vert2.max(0)) 
    
    faces = mesh.faces 
    tri = torch.from_numpy(faces.astype(np.int32)).int().cuda()
    pos_or =torch.from_numpy(vert2).float().cuda().unsqueeze(0)
    
    batch_size = pos_or.shape[0]
    col = torch.ones((pos_or.shape[1], 3), dtype=torch.float32).cuda()
    
    mv = torch.eye(4).cuda().float()
    # mv[1, 3] += 0.3 
    mv = mv.unsqueeze(0).repeat(batch_size, 1,1) 

    clip_features = torch.zeros(batch_size, 8, 512).cuda()  
    dino_features = torch.zeros(batch_size, 8, 384).cuda()   
    
    for angle_i in range(8):
        alpha_y = torch.tensor(2*torch.pi * angle_i /8)
        rot_y = torch.tensor([[torch.cos(alpha_y), 0, torch.sin(alpha_y)], [0, 1, 0], [-torch.sin(alpha_y), 0, torch.cos(alpha_y)]])
        rot_y = rot_y.unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pos = torch.matmul(pos_or, rot_y)
        with torch.no_grad():
            color, depth = vertex_color_render(pos, tri, col, uv=None, mv=mv, proj=None, image_size=(256, 256))
        
        clip_images = torch.zeros(batch_size, 3, 224, 224).cuda()  
        dino_images = torch.zeros(batch_size, 3, 224, 224).cuda() 
        
        for j in range(batch_size):
            img_or = depth.cpu().numpy()[j] 
            img = img_or + 1
            img = img / 2
            img = 1 - img 
            img = img * (1-(img_or==0).astype(np.float32)) 
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            tmp_path = os.path.join(out_path_im, '%d.png'%(angle_i,)) 
            imageio.imsave(tmp_path, img)
            image = clip_preprocess(Image.open(tmp_path)).unsqueeze(0).to(device)
            clip_images[j] = image 
            image_dino = dinov2_preprocess(Image.open(tmp_path).convert("RGB")).unsqueeze(0).to(device) 
            dino_images[j] = image_dino 
            
        with torch.no_grad():
            clip_features[:, angle_i] = clip_model.encode_image(clip_images)
            dino_features[:, angle_i] = dinov2_vits14(dino_images)
    # print(shape_param.shape)
    # print(clip_features.shape, dino_features.shape)       
    shape_param = np.zeros(1)      
    np.savez(os.path.join(out_path, "tmp.npz"), shape_param=shape_param, 
                 clip_features =clip_features.cpu().numpy(),  dino_features=dino_features.cpu().numpy(),)
    print('saved' )
    
if __name__ == '__main__':
    # main()
    render_one_mesh()
