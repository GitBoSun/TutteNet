import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt
from eval_utils.eval_utils import eval_one_epoch

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, freeze_modules_except
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

# os.environ['CUDA_VISIBLE_DEVICES']="4,5,6,7"

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

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None


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

    model = build_network(model_cfg=cfg.MODEL, cfg=cfg, dataset=train_set)

    if args.refinement:
        model = freeze_modules_except(model, 'refinement')
        model.refinement_enabled = True

    rank = cfg.LOCAL_RANK % torch.cuda.device_count()
    if rank == 0:
        total_model_size = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                total_model_size += p.numel()
        logger.info(f"MODEL SIZE (N. Parameters)={total_model_size/2**20} M")

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    torch.autograd.set_detect_anomaly(True)


    cfg.OPTIMIZATION['BATCH_SIZE_PER_GPU'] *= 1
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=args.find_unused_parameters)
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    if args.eval_with_train:
        test_set, test_loader, test_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger, training=False
        )
        eval_with_train=[
            cfg, args, dist_train, logger, output_dir, ckpt_dir, eval_one_epoch, test_loader
        ]
    else:
        eval_with_train = None
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        eval_with_train=eval_with_train,
        logger=logger,
        eval_interval=args.eval_interval,
        train_interval=args.train_interval,
        vis_prefix=args.vis_prefix, 
        save_train=args.save_train, 
    )

    #if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
    #    train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if not args.eval_with_train:
        logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers, logger=logger, training=False
        )
        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

        repeat_eval_ckpt(
            model.module if dist_train else model,
            test_loader, args, eval_output_dir, logger, ckpt_dir,
            dist_test=dist_train
        )
        logger.info('**********************End evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
