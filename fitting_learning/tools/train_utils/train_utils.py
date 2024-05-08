import glob
import os
import re

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils, vis_utils
from pcdet.models import freeze_modules, freeze_modules_except
from pcdet.models import load_data_to_gpu

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, logger=None, cur_epoch=0,
                    results_dir=None, train_interval=10):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    num_gpus = torch.cuda.device_count()

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()

    for cur_it in range(total_it_each_epoch):

        end = time.time()
        try:
            batch = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            
        model.train()
        optimizer.zero_grad()
        
        # load_data_to_gpu(batch)
        # pred_dicts, loss = model(batch)
        
        loss, tb_dict, disp_dict, pred_dicts = model_func(model, batch)
        
        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer
        
        loss.backward()
        # loss.backward(retain_graph=True)
        if optim_cfg.GRAD_NORM_CLIP > 0:
            grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        else:
            grad_norm = 0.0
        
        if rank == 0:
            if 1:
            # if tb_log is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print("WWWWW", name)
                    
                    continue
                    if not param.requires_grad:
                        continue
                    tb_log.add_scalar(f'grad_linf/{name}', param.grad.abs().max().item(), accumulated_iter)
                    tb_log.add_scalar(f'grad_l2/{name}', param.grad.norm(p='fro').item(), accumulated_iter)

        zerograd_modules = optim_cfg.get('ZEROGRAD_MODULES', None)
        if zerograd_modules is not None:
            for name, param in model.named_parameters():
                for module_regex, start_iter in zerograd_modules.items():
                    if start_iter > accumulated_iter:
                        if re.match(module_regex, name) is not None:
                            if param.grad is not None:
                                print('name', name)
                                param.grad[:] = 0

        optimizer.step()
        lr_scheduler.step(accumulated_iter // (num_gpus * batch['batch_size']))
        accumulated_iter += num_gpus * batch['batch_size']
        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            #model.module.update_ema()
            #model.update_ema()
            # import pdb; pdb.set_trace()
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({'loss': loss.item()})
            if 'j_loss' in tb_dict.keys():
                disp_dict.update({'loss_point': tb_dict['reg_loss'], 'loss_j': tb_dict['j_loss'], })
            if 'm_loss' in tb_dict.keys():
                disp_dict.update({ 'loss_m': tb_dict['m_loss']})
            if 'd_loss' in tb_dict.keys():
                disp_dict.update({'loss_point': tb_dict['reg_loss'], 'loss_d': tb_dict['d_loss'], 
                                  'loss_r': tb_dict['r_loss'], })
            
            disp_dict.update({
                'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                'grad': f'{grad_norm:.4f}'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('train/grad_norm', grad_norm, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                tb_log.add_scalar('meta_data/batch_time', batch_time.avg, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            if cur_it%100==0:
                if 'm_loss' in tb_dict.keys():
                    logger.info('[Epoch %d] [It %d/%d]: loss: %.6f, loss_pt: %.6f, loss_j: %.4f, loss_m: %.4f, loss_d: %.4f, loss_r: %.4f, grad_norm: %.4f, lr: %.6f'%(cur_epoch, cur_it, total_it_each_epoch, \
                        loss.item(), tb_dict['reg_loss'], tb_dict['j_loss'], tb_dict['m_loss'],tb_dict['d_loss'], tb_dict['r_loss'], grad_norm, cur_lr))
                elif 'j_loss' in tb_dict.keys():
                    logger.info('[Epoch %d] [It %d/%d]: loss: %.6f, loss_pt: %.6f, loss_j: %.4f, loss_d: %.4f, loss_r: %.4f, grad_norm: %.4f, lr: %.6f'%(cur_epoch, cur_it, total_it_each_epoch, \
                        loss.item(), tb_dict['reg_loss'], tb_dict['j_loss'], tb_dict['d_loss'], tb_dict['r_loss'], grad_norm, cur_lr))
                elif 'd_loss' in tb_dict.keys():
                    logger.info('[Epoch %d] [It %d/%d]: loss: %.6f, loss_pt: %.6f, loss_d: %.4f, loss_r: %.4f, grad_norm: %.4f, lr: %.6f'%(cur_epoch, cur_it, total_it_each_epoch, \
                        loss.item(), tb_dict['reg_loss'], tb_dict['d_loss'], tb_dict['r_loss'], grad_norm, cur_lr))
                else:
                    logger.info('[Epoch %d] [It %d/%d]: loss: %.6f, grad_norm: %.4f, lr: %.6f'%(cur_epoch, cur_it, total_it_each_epoch, \
                        loss.item(), grad_norm, cur_lr))

                if results_dir is not None and cur_epoch % train_interval==0 and cur_it%1000==0:
                    vis_utils.plot_embed_results(pred_dicts, results_dir, epoch=cur_epoch)

    if rank == 0:
        pbar.close()

    return accumulated_iter

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, eval_with_train=None, logger=None, eval_interval=10, train_interval=10,
                vis_prefix="", save_train=True):
    accumulated_iter = start_iter

    if eval_with_train is not None:
        cfg, args, dist_train, logger, output_dir, ckpt_dir, \
            eval_one_epoch, test_loader = eval_with_train
        logger.info('**********************Evaluation with training %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    
   
        # args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs
    train_output_dir = output_dir / 'train'
    train_output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            if train_loader.dataset.data_augmentor is not None:
                train_loader.dataset.data_augmentor.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                logger=logger,
                cur_epoch=cur_epoch,
                results_dir=train_output_dir,
                train_interval=train_interval,
            )

            # save trained model
            
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0 and save_train:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
               )

            if eval_with_train is not None:
                if cur_epoch % eval_interval == 0 or cur_epoch==total_epochs-1:
                # if (args.epochs - cur_epoch <= args.num_epochs_to_eval):
                    ret_dict = eval_one_epoch(
                        cfg,
                        model.module if dist_train else model,
                        test_loader, cur_epoch, logger, dist_test=dist_train,
                        result_dir=eval_output_dir,
                        vis_prefix=vis_prefix, 
                    )

                    if tb_log is not None:
                        for key, val in ret_dict.items():
                            tb_log.add_scalar('eval/' + key, val, accumulated_iter)

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())

            if model.module.ema is not None:
                model_state_ema = model_state_to_cpu(model.module.ema)
            model_state['ema'] = model_state_ema
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
