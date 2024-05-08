import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        if isinstance(info[key], np.ndarray):
            ret_info[key] = info[key][keep_indices]
        elif isinstance(info[key], list):
            ret_info[key] = [info[key][k] for k in keep_indices]
    return ret_info


def apply_to_dict(data_dict, func):
    ret_data_dict = {}
    for key in data_dict.keys():
        ret_data_dict[key] = func(data_dict[key])
    return ret_data_dict


def transform_name(data_dict, func):
    ret_data_dict = {}
    for key in data_dict.keys():
        ret_data_dict[func(key)] = data_dict[key]
    return ret_data_dict


# def filter_dict(data_dict, mask, ignore_keys = ['point_velo']):
def filter_dict(data_dict, mask, ignore_keys = []):
    ret_data_dict = {}
    for key in data_dict.keys():
        if key in ignore_keys:
            ret_data_dict[key] = data_dict[key]
            continue
        if isinstance(mask, torch.Tensor) and (mask.dtype == torch.bool):
            assert mask.shape[0] == len(data_dict[key]), f"MisMatch for key={key}, mask.shape={mask.shape}, data.shape={len(data_dict[key])}"
        if isinstance(mask, np.ndarray) and (mask.dtype == bool):
            assert mask.shape[0] == len(data_dict[key]), f"MisMatch for key={key}, mask.shape={mask.shape}, data.shape={len(data_dict[key])}"
        ret_data_dict[key] = data_dict[key][mask]
    return ret_data_dict

def indexing_list_elements(data_dict, idx):
    if data_dict is None:
        return None
    ret_data_dict = {}
    for key in data_dict.keys():
        if isinstance(data_dict[key], list):
            ret_data_dict[key] = data_dict[key][idx]
        else:
            ret_data_dict[key] = data_dict[key]
    return ret_data_dict

def concat_dicts(data_dicts):
    ret_data_dict = defaultdict(list)
    for data_dict in data_dicts:
        for key in data_dict.keys():
            ret_data_dict[key].append(data_dict[key])
    for key in ret_data_dict.keys():
        ret_data_dict[key] = np.concatenate(ret_data_dict[key], axis=0)
    return ret_data_dict

def torch_concat_dicts(data_dicts):
    ret_data_dict = defaultdict(list)
    for data_dict in data_dicts:
        for key in data_dict.keys():
            ret_data_dict[key].append(data_dict[key])
    for key in ret_data_dict.keys():
        ret_data_dict[key] = torch.cat(ret_data_dict[key], dim=0)
    return ret_data_dict

def stack_dicts(data_dicts, pad_to_size=None):
    ret_data_dict = defaultdict(list)
    for data_dict in data_dicts:
        for key in data_dict.keys():
            if pad_to_size is not None:
                if data_dict[key].shape[0] < pad_to_size:
                    pad_data = np.zeros((pad_to_size-data_dict[key].shape[0], *data_dict[key].shape[1:]),
                                        dtype=data_dict[key].dtype)
                    data_dict[key] = np.concatenate([data_dict[key], pad_data], axis=0)
            ret_data_dict[key].append(data_dict[key])
    for key in ret_data_dict.keys():
        ret_data_dict[key] = np.stack(ret_data_dict[key], axis=0)
    return ret_data_dict


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def rotate_points_along_z_np(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """

    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0]).astype(points.dtype)
    ones = np.ones(points.shape[0]).astype(angle.dtype)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3).astype(np.float32)
    points_rot = points[:, :, 0:3] @ rot_matrix
    points_rot = np.concatenate([points_rot, points[:, :, 3:]], axis=-1)
    return points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def get_voxel_corners(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_corners = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_corners.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_corners.device).float()
    voxel_corners = (voxel_corners) * voxel_size + pc_range
    return voxel_corners

def get_cylinder_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    def polar2cat(input_xyz_polar):
        x = input_xyz_polar[:, 0] * torch.cos(input_xyz_polar[:, 1])
        y = input_xyz_polar[:, 0] * torch.sin(input_xyz_polar[:, 1])
        return torch.stack((x, y, input_xyz_polar[:, 2]), dim=1)

    assert voxel_coords.shape[1] == 3
    # voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_centers = voxel_coords.float() 
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    voxel_centers = polar2cat(voxel_centers)
    return voxel_centers

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    #if mp.get_start_method(allow_none=True) is None:
    #    mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
