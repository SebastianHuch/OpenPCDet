import glob
import os
import csv
import pickle

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from tools.eval_utils import eval_utils


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

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

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict = model_func(model, batch)

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,
                cfg=None, val_loader=None, logger=None, dist_train=False, eval_output_dir=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        with open(os.path.join(eval_output_dir, "results.csv"), "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["epoch_id",
                                 "val_loss_avg",
                                 "AP_Car_3d/0.5_R11",
                                 "AP_Car_3d/0.5_R40",
                                 "AP_Car_3d/0.7_R11",
                                 "AP_Car_3d/0.7_R40",
                                 "recall/rcnn_0.3",
                                 "recall/rcnn_0.5",
                                 "recall/rcnn_0.7",
                                 "recall/roi_0.3",
                                 "recall/roi_0.5",
                                 "recall/roi_0.7",
                                 "avg_pred_obj"])

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            logger.info(f'Currently running epoch: {cur_epoch}')
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

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
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # Evaluate epoch
            model.eval()
            model.eval_mode = True
            for module in model.module_list:
                if hasattr(module, "eval_mode"):
                    module.eval_mode = True
            for tag, parm in model.named_parameters():
                tb_log.add_histogram(tag, parm.data.cpu().numpy(), cur_epoch)

            cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch) / cfg.DATA_CONFIG.DATA_SPLIT['test']
            ret_dict, val_loss_list = eval_utils.eval_one_epoch(
                cfg, model, val_loader, epoch_id=cur_epoch, logger=logger, dist_test=dist_train,
                result_dir=cur_result_dir, get_val_loss=False
            )
            if len(val_loss_list) > 0:
                val_losses = [loss[0].item() for loss in val_loss_list]
                val_loss_avg = sum(val_losses) / len(val_losses)
                tb_log.add_scalar('eval_during_train/val_loss', val_loss_avg, cur_epoch)
            result_str = ""
            for key in sorted(ret_dict):
                tb_log.add_scalar(f"eval_during_train/{key}", ret_dict[key], cur_epoch)
                result_str += str(ret_dict[key]) + ";"
            result_str = result_str[:-1]
            model.train()
            model.eval_mode = False
            for module in model.module_list:
                if hasattr(module, "eval_mode"):
                    module.eval_mode = False

            # Save results to CSV
            if os.path.getsize(cur_result_dir / 'result.pkl') > 0:
                with open(cur_result_dir / 'result.pkl', 'rb') as f:
                    det_annos = pickle.load(f)
                    total_pred_objects = 0
                    for anno in det_annos:
                        total_pred_objects += anno['name'].__len__()
                    avg_pred_obj = total_pred_objects / max(1, len(det_annos))
            else:
                # results.pkl empty, no predictions
                avg_pred_obj = 0.0
            if len(val_loss_list) > 0:
                with open(os.path.join(eval_output_dir, "results.csv"), "a") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([cur_epoch, val_loss_avg, result_str, avg_pred_obj])


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
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
