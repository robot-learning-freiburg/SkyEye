import os
import sys

import argparse
import shutil
import time
import json
from collections import OrderedDict
from os import path
import numpy as np
import wandb
import random
import tensorboardX as tensorboard
import torch
import torch.optim as optim
import torch.utils.data as data
from torch import distributed
from inplace_abn import ABN

from skyeye.config.config import load_config
from skyeye.utils.sequence import pad_packed_images

from skyeye.utils.visualisation import generate_visualisations, plot_confusion_matrix

from skyeye.data.dataset import BEVKitti360Dataset
from skyeye.data.transform import BEVTransform
from skyeye.data.misc import iss_collate_fn
from skyeye.data.sampler import DistributedARBatchSampler

from skyeye.modules.voxel_grid import DenseVoxelGrid
from skyeye.modules.heads.semantic import FVSemHead

from skyeye.models.backbone_edet.efficientdet_small import EfficientDet
from skyeye.models.unsupervised_bev import UnsupervisedBevNet, NETWORK_INPUTS_FV

from skyeye.algos.voxel_grid import VoxelGridAlgo
from skyeye.algos.semantic_seg import SemanticSegLoss, SemanticSegAlgo

from skyeye.utils import logging
from skyeye.utils.meters import AverageMeter, ConfusionMatrixMeter
from skyeye.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, all_reduce_losses
from skyeye.utils.parallel import DistributedDataParallel
from skyeye.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots

parser = argparse.ArgumentParser(description="BEV Unsupervised training script")
parser.add_argument("--run_name", required=True, type=str, help="Name of the run")
parser.add_argument("--project_root_dir", required=True, type=str, help="The root directory of the project")
parser.add_argument("--seam_root_dir", required=True, type=str, help="The root directory of the data")
parser.add_argument("--dataset_root_dir", required=True, type=str,
                    help="The root directory of the data from which the Seam format was generated")
parser.add_argument("--mode", required=True, type=str, help="'train' the model or 'test' the model?")
parser.add_argument("--train_dataset", type=str, default="Kitti360", help="Name of the dataset to be used for training")
parser.add_argument("--val_dataset", type=str, default="Kitti360", help="Name of the dataset to be used for validation")
parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file", nargs="?")
parser.add_argument("--pre_train", type=str, nargs="*",
                    help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                         "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                         "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                         "will be loaded from the snapshot")
parser.add_argument("--defaults_config", type=str, help="Path to defaults configuration file")
parser.add_argument("--config", required=True, type=str, help="Path to configuration file")
parser.add_argument("--comment", type=str, help="Comment to add to WandB")
parser.add_argument("--debug", type=bool, default=False, help="Should the program run in 'debug' mode?")
parser.add_argument("--freeze_modules", nargs='+', default=[], help="The modules to freeze. Default is empty")
parser.add_argument("--use_wandb", type=bool, default=False)

def make_config(args, config_file, defaults_config_file=None):
    if defaults_config_file is not None:
        logging.log_info("Loading default configuration from %s", defaults_config_file, debug=args.debug)
    logging.log_info("Loading overriding configuration from %s", config_file, debug=args.debug)
    conf = load_config(config_file, defaults_config_file)

    logging.log_info("\n%s", config_to_string(conf), debug=args.debug)
    return conf


def create_run_directories(args, rank):
    root_dir = args.project_root_dir
    experiment_dir = os.path.join(root_dir, "experiments")
    if args.mode == "train":
        run_dir = os.path.join(experiment_dir, "skyeye_fv_train_{}".format(args.run_name))
    elif args.mode == "test":
        run_dir = os.path.join(experiment_dir, "skyeye_fv_test_{}".format(args.run_name))
    else:
        raise RuntimeError("Invalid choice. --mode must be either 'train' or 'test'")

    saved_models_dir = os.path.join(run_dir, "saved_models")
    log_dir = os.path.join(run_dir, "logs")
    config_file = os.path.join(run_dir, args.config)
    if args.defaults_config is not None:
        defaults_config_file = os.path.join(run_dir, "defaults_{}".format(args.defaults_config))
    else:
        defaults_config_file = None

    # Create the directory
    if rank == 0 and (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
    if rank == 0:
        assert not os.path.exists(run_dir), "Run folder already found! Delete it to reuse the run name."

    if rank == 0:
        os.mkdir(run_dir)
        os.mkdir(saved_models_dir)
        os.mkdir(log_dir)

    # Copy the config file into the folder
    config_path = os.path.join(experiment_dir, "config", args.config)
    defaults_config_path = os.path.join(experiment_dir, "config", "defaults", args.defaults_config)
    if rank == 0:
        shutil.copyfile(config_path, config_file)
        if defaults_config_file is not None:
            shutil.copyfile(defaults_config_path, defaults_config_file)

    return log_dir, run_dir, saved_models_dir, config_path, defaults_config_path


def make_dataloader(args, config, rank, world_size):
    dl_config = config['dataloader']

    logging.log_info("Creating train dataloader for {} dataset".format(args.train_dataset), debug=args.debug)
    logging.log_info("Creating val dataloader for {} dataset".format(args.val_dataset), debug=args.debug)

    # Train dataloaders
    train_tf = BEVTransform(shortest_size=dl_config.getint("shortest_size"),
                            longest_max_size=dl_config.getint("longest_max_size"),
                            rgb_mean=dl_config.getstruct("rgb_mean"),
                            rgb_std=dl_config.getstruct("rgb_std"),
                            front_resize=dl_config.getstruct("front_resize"),
                            bev_crop=dl_config.getstruct("bev_crop"),
                            scale=dl_config.getstruct("scale"),
                            random_flip=dl_config.getboolean("random_flip"),
                            random_brightness=dl_config.getstruct("random_brightness"),
                            random_contrast=dl_config.getstruct("random_contrast"),
                            random_saturation=dl_config.getstruct("random_saturation"),
                            random_hue=dl_config.getstruct("random_hue"))

    if args.train_dataset == "Kitti360":
        train_db = BEVKitti360Dataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                      split_name=dl_config['train_set'], transform=train_tf,
                                      window=dl_config.getint('fvsem_window_size'))

    if not args.debug:
        train_sampler = DistributedARBatchSampler(train_db, dl_config.getint('train_batch_size'), world_size, rank, True)
        train_dl = torch.utils.data.DataLoader(train_db,
                                               batch_sampler=train_sampler,
                                               collate_fn=iss_collate_fn,
                                               pin_memory=True,
                                               num_workers=dl_config.getint("train_workers"))
    else:
        train_dl = torch.utils.data.DataLoader(train_db,
                                               batch_size=dl_config.getint('train_batch_size'),
                                               shuffle=True,
                                               collate_fn=iss_collate_fn,
                                               pin_memory=True,
                                               num_workers=dl_config.getint("train_workers"))

    # Validation datalaader
    val_tf = BEVTransform(shortest_size=dl_config.getint("shortest_size"),
                          longest_max_size=dl_config.getint("longest_max_size"),
                          rgb_mean=dl_config.getstruct("rgb_mean"),
                          rgb_std=dl_config.getstruct("rgb_std"),
                          front_resize=dl_config.getstruct("front_resize"),
                          bev_crop=dl_config.getstruct("bev_crop"))

    if args.val_dataset == "Kitti360":
        val_db = BEVKitti360Dataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                    split_name=dl_config['val_set'], transform=val_tf, window=0)

    if not args.debug:
        val_sampler = DistributedARBatchSampler(val_db, dl_config.getint("val_batch_size"), world_size, rank, False)
        val_dl = torch.utils.data.DataLoader(val_db,
                                             batch_sampler=val_sampler,
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))
    else:
        val_dl = torch.utils.data.DataLoader(val_db,
                                             batch_size=dl_config.getint("val_batch_size"),
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))

    return train_dl, val_dl


def make_model(args, config, fv_num_thing, fv_num_stuff, bev_num_thing, bev_num_stuff):
    base_config = config["base"]
    voxel_config = config['voxelgrid']
    dataset_config = config['dataset']
    sem_config = config['sem']
    cam_config = config['cameras']
    dl_config = config['dataloader']

    fv_classes = {"total": fv_num_thing + fv_num_stuff, "stuff": fv_num_stuff, "thing": fv_num_thing}
    bev_classes = {"total": bev_num_thing + bev_num_stuff, "stuff": bev_num_stuff, "thing": bev_num_thing}

    # BN + activation
    if not args.debug:
        norm_act_2d, norm_act_3d, norm_act_static = norm_act_from_config(base_config)
    else:
        norm_act_2d, norm_act_3d, norm_act_static = ABN, ABN, ABN

    # Create the backbone
    model_compount_coeff = int(base_config["base"][-1])
    model_name = "efficientdet-d{}".format(model_compount_coeff)
    logging.log_info("Creating backbone model %s", base_config["base"], debug=args.debug)
    body = EfficientDet(compound_coef=model_compount_coeff)
    ignore_layers = []
    body = EfficientDet.from_pretrained(body, model_name, ignore_layers=ignore_layers)

    bev_params = cam_config.getstruct('bev_params')
    resolution = bev_params['cam_z'] / bev_params['f'] / dl_config.getfloat("scale")

    # Create the Dense Voxel Grid
    voxel_grid = DenseVoxelGrid(in_channels=voxel_config.getstruct("in_channels")[base_config["base"]],
                                bev_params=bev_params, feat_scale=voxel_config.getfloat("in_feat_scale"),
                                bev_W_out=dl_config.getstruct("bev_crop")[0] * dl_config.getfloat("scale"),
                                bev_Z_out=dl_config.getstruct("bev_crop")[1] * dl_config.getfloat("scale"),
                                y_extents=voxel_config.getstruct("y_extents"),
                                norm_act_2d=norm_act_2d,
                                norm_act_3d=norm_act_3d)

    voxel_grid_algo = VoxelGridAlgo(feat_scale=voxel_config.getfloat("in_feat_scale"),
                                    resolution=resolution,
                                    y_extents=voxel_config.getstruct("y_extents"))

    # Create the FV semantic segmentation network
    fv_sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("fv_ohem"),
                                  class_weights=sem_config.getstruct("fv_class_weights"))
    fv_sem_algo = SemanticSegAlgo(fv_sem_loss, fv_classes["total"])
    fv_sem_head = FVSemHead(num_classes=fv_classes['total'])

    return UnsupervisedBevNet(body, voxel_grid, fv_sem_head, None, voxel_grid_algo, fv_sem_algo, None,
                              dataset=args.train_dataset, fv_classes=fv_classes, bev_classes=bev_classes,
                              fv_sky_index=dataset_config.getint("fv_sky_index"),
                              fv_veg_index=dataset_config.getint("fv_veg_index"))


def make_optimizer(config, model, epoch_length):
    opt_config = config["optimizer"]
    sch_config = config["scheduler"]

    if opt_config['opt_algo'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=opt_config.getfloat("base_lr"),
                               weight_decay=opt_config.getfloat("weight_decay"))
    elif opt_config['opt_algo'] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=opt_config.getfloat("base_lr"),
                              weight_decay=opt_config.getfloat("weight_decay"))
    else:
        raise NotImplementedError()

    scheduler = scheduler_from_config(sch_config, optimizer, epoch_length)

    assert sch_config["update_mode"] in ("batch", "epoch")
    batch_update = sch_config["update_mode"] == "batch"
    total_epochs = sch_config.getint("epochs")

    return optimizer, scheduler, batch_update, total_epochs


def freeze_modules(args, model):
    for module in args.freeze_modules:
        print("Freezing module: {}".format(module))
        for name, param in model.named_parameters():
            if name.startswith(module):
                param.requires_grad = False

    return model


def train(model, optimizer, grad_scaler, scheduler, dataloader, meters, **varargs):
    model.train()
    if not varargs['debug']:
        dataloader.batch_sampler.set_epoch(varargs["epoch"])
    optimizer.zero_grad()

    global_step = varargs["global_step"]
    loss_weights = varargs['loss_weights']

    time_meters = {"data_time": AverageMeter((), meters["loss"].momentum),
                   "batch_time": AverageMeter((), meters["loss"].momentum)}

    data_time = time.time()

    for it, sample in enumerate(dataloader):
        sample_cuda = {}

        for key in NETWORK_INPUTS_FV:
            if isinstance(sample[key], list):
                sample_cuda[key] = [m.cuda(device=varargs['device'], non_blocking=True) for m in sample[key] if
                                    not isinstance(m, dict)]
            else:
                sample_cuda[key] = sample[key].cuda()

        # Get the intrinsics from a fixed location or from the dataloader based on the dataset
        sample_cuda['fv_intrinsics'] = [pad_packed_images(intrinsics)[0] for intrinsics in sample_cuda['fv_intrinsics']]
        sample_cuda['ego_pose'] = [pad_packed_images(pose)[0] for pose in sample_cuda['ego_pose']]
        sample_cuda['fvsem_window_size'] = varargs['fvsem_window_size']
        sample_cuda['fvsem_step_size'] = varargs['fvsem_step_size']
        sample_cuda['transform_status'] = sample['transform_status']
        del sample

        # Log the data loading time
        time_meters['data_time'].update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        # Run network with automatic mixed precision
        with torch.cuda.amp.autocast():
            losses, results, stats = model(**sample_cuda, do_loss=True, use_fv=True, use_bev=False)

            if not varargs['debug']:
                distributed.barrier()

            losses = OrderedDict((k, v.mean()) for k, v in losses.items())
            losses["loss"] = sum(loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys())

        # Increment the optimiser and backpropagate the gradients
        grad_scaler.scale(losses['loss']).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

        # Log the time taken to execute the batch
        time_meters['batch_time'].update(torch.tensor(time.time() - batch_time))

        # Gather stats from all workers
        if not varargs['debug']:
            losses = all_reduce_losses(losses)

        fv_sem_conf_stat = stats['fv_sem_conf']

        # Gather the stats from all the workers
        if not varargs['debug']:
            distributed.all_reduce(fv_sem_conf_stat, distributed.ReduceOp.SUM)

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())
            meters['fv_sem_conf'].update(fv_sem_conf_stat.cpu())

        # Clean-up
        del losses, stats, sample_cuda

        # Log
        if (it + 1) % varargs["log_interval"] == 0:
            if varargs["summary"] is not None:
                logging.log_iter("train", meters, time_meters, None, batch=True, global_step=global_step,
                                 epoch=varargs["epoch"], num_epochs=varargs['num_epochs'],
                                 lr=scheduler.get_lr()[0], curr_iter=it + 1, num_iters=len(dataloader),
                                 summary=varargs['summary'], debug=varargs['debug'])
            if varargs['wandb_summary'] is not None:
                logging.log_wandb(varargs['wandb_summary'], "train", meters, time_meters, metrics=None, batch=True,
                                  epoch=varargs["epoch"], lr=scheduler.get_lr()[0], global_step=global_step,
                                  debug=varargs['debug'])

        data_time = time.time()

    if varargs['wandb_summary'] is not None:
        logging.log_wandb(varargs['wandb_summary'], "train", meters, time_meters, metrics=None, batch=False,
                          epoch=varargs["epoch"], lr=scheduler.get_lr()[0], global_step=global_step,
                          debug=varargs['debug'])

    del results
    return global_step


def validate(model, dataloader, **varargs):
    model.eval()

    if not varargs['debug']:
        dataloader.batch_sampler.set_epoch(varargs["epoch"])

    fv_num_classes = dataloader.dataset.fv_num_stuff + dataloader.dataset.fv_num_thing
    loss_weights = varargs['loss_weights']

    val_meters = {
        "loss": AverageMeter(()),
        "fv_sem_loss": AverageMeter(()),
        "fv_sem_conf": ConfusionMatrixMeter(fv_num_classes),
    }

    time_meters = {"data_time": AverageMeter(()),
                   "batch_time": AverageMeter(())}

    # Validation metrics
    val_metrics = {"fv_sem_miou": AverageMeter(())}

    # Accumulators for AP, mIoU and panoptic computation
    fv_semantic_conf_mat = torch.zeros(fv_num_classes, fv_num_classes, dtype=torch.double)

    data_time = time.time()
    wandb_vis_dict = {}
    max_vis_count = 5

    for it, sample in enumerate(dataloader):

        idxs = sample['idx']
        with torch.no_grad():
            sample_cuda = {}
            for key in NETWORK_INPUTS_FV:
                if isinstance(sample[key], list):
                    sample_cuda[key] = [m.cuda(device=varargs['device'], non_blocking=True) for m in sample[key] if
                                        not isinstance(m, dict)]
                else:
                    sample_cuda[key] = sample[key].cuda()

            # Get the intrinsics from a fixed location or from the dataloader based on the dataset
            sample_cuda['fv_intrinsics'] = [pad_packed_images(intrinsics)[0] for intrinsics in sample_cuda['fv_intrinsics']]
            sample_cuda['ego_pose'] = [pad_packed_images(pose)[0] for pose in sample_cuda['ego_pose']]
            sample_cuda['fvsem_window_size'] = varargs['fvsem_window_size']
            sample_cuda['fvsem_step_size'] = varargs['fvsem_step_size']
            sample_cuda['transform_status'] = sample['transform_status']
            del sample

            time_meters['data_time'].update(torch.tensor(time.time() - data_time))
            batch_time = time.time()

            # Run network
            losses, results, stats = model(**sample_cuda, do_loss=True, use_fv=True, use_bev=False)

            if not varargs['debug']:
                distributed.barrier()

            losses = OrderedDict((k, v.mean()) for k, v in losses.items())
            losses["loss"] = sum(loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys())

            time_meters['batch_time'].update(torch.tensor(time.time() - batch_time))

            # Separate the normal and abnormal stats entries
            fv_sem_conf_stat = stats['fv_sem_conf']
            rem_stats = {k: v for k, v in stats.items() if (not k.endswith("sem_conf"))}
            if not varargs['debug']:
                distributed.all_reduce(fv_sem_conf_stat, distributed.ReduceOp.SUM)

            # Add the semantic confusion matrix to the existing one
            fv_semantic_conf_mat += fv_sem_conf_stat.cpu()

            # Update meters
            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    val_meters[loss_name].update(loss_value.cpu())
                for stat_name, stat_value in rem_stats.items():
                    val_meters[stat_name].update(stat_value.cpu())
                val_meters['fv_sem_conf'].update(fv_sem_conf_stat.cpu())

            del losses, stats

            # Do the post-processing
            count_keys = [key for key in wandb_vis_dict.keys() if key.startswith("count")]
            get_vis = (it + 1) % varargs['log_interval'] == 0 and \
                      (len(count_keys) == 0 or wandb_vis_dict[count_keys[0]] < max_vis_count)

            # Get the visualisation
            if get_vis:
                wandb_vis_dict = generate_visualisations(sample_cuda, results, idxs, wandb_vis_dict,
                                                         img_scale=varargs['img_scale'],
                                                         fv_num_stuff=dataloader.dataset.fv_num_stuff,
                                                         bev_num_stuff=dataloader.dataset.bev_num_stuff,
                                                         dataset=dataloader.dataset.dataset_name,
                                                         fvsem_window_size=varargs['fvsem_window_size'],
                                                         fvsem_step_size=varargs['fvsem_step_size'],
                                                         rgb_mean=varargs['rgb_mean'], rgb_std=varargs['rgb_std'],
                                                         max_vis_count=max_vis_count)

            # Log batch
            if (it + 1) % varargs["log_interval"] == 0:
                if varargs['summary'] is not None:
                    logging.log_iter("val", val_meters, time_meters, None, global_step=varargs['global_step'],
                                     epoch=varargs['epoch'], num_epochs=varargs['num_epochs'], lr=None, curr_iter=it+1,
                                     num_iters=len(dataloader), summary=None, debug=varargs['debug'])

                if varargs['wandb_summary'] is not None:
                    logging.log_wandb(varargs['wandb_summary'], "val", val_meters, time_meters, None, batch=True,
                                      epoch=varargs["epoch"], lr=None, global_step=varargs['global_step'],
                                      debug=varargs['debug'])

            data_time = time.time()

    # Finalise FV semantic mIoU computation
    fv_semantic_conf_mat = fv_semantic_conf_mat.to(device=varargs['device'])
    if not varargs['debug']:
        distributed.all_reduce(fv_semantic_conf_mat, distributed.ReduceOp.SUM)
    fv_semantic_conf_mat = fv_semantic_conf_mat.cpu()[:fv_num_classes, :]
    fv_sem_intersection = fv_semantic_conf_mat.diag()
    fv_sem_union = ((fv_semantic_conf_mat.sum(dim=1) + fv_semantic_conf_mat.sum(dim=0) - fv_semantic_conf_mat.diag()) + 1e-8)
    fv_sem_miou = fv_sem_intersection / fv_sem_union

    # Plot the confusion matrix
    fv_sem_conf_mat_plt = plot_confusion_matrix(fv_semantic_conf_mat, fv_num_classes,
                                                dataset=dataloader.dataset.dataset_name, bev=False)
    wandb_vis_dict['wandb_fv_sem_conf'] = wandb.Image(fv_sem_conf_mat_plt)

    # Save the metrics
    scores = {}
    scores['fv_sem_miou'] = fv_sem_miou.mean()

    # Update the validation metrics meters
    for key in val_metrics.keys():
        if key in scores.keys():
            if scores[key] is not None:
                val_metrics[key].update(scores[key].cpu())

    # Log the accumulated wandb images and the confusion_matrix
    if not varargs['debug'] and varargs['wandb_summary'] is not None:
        wandb_keys = [key for key in wandb_vis_dict.keys() if key.startswith("wandb")]
        for key in wandb_keys:
            varargs['wandb_summary'].log({"val_samples/{}".format(key): wandb_vis_dict[key]}, step=varargs['global_step'])

    # To prevent overwriting
    del fv_sem_conf_mat_plt, wandb_vis_dict

    # Log results
    logging.log_info("Validation done", debug=varargs['debug'])
    if varargs["summary"] is not None:
        logging.log_iter("val", val_meters, time_meters, val_metrics, batch=False, summary=varargs['summary'],
                         global_step=varargs['global_step'], curr_iter=len(dataloader), num_iters=len(dataloader),
                         epoch=varargs['epoch'], num_epochs=varargs['num_epochs'], lr=None, debug=varargs['debug'])
    if varargs['wandb_summary'] is not None:
        logging.log_wandb(varargs['wandb_summary'], "val", val_meters, time_meters, val_metrics, batch=False,
                          epoch=varargs['epoch'], lr=None, global_step=varargs['global_step'], debug=varargs['debug'])

    fv_class_names = [name for idx, name in enumerate(dataloader.dataset.fv_categories)]
    logging.log_miou("FV Sem mIoU", fv_sem_miou, fv_class_names)

    return scores['fv_sem_miou'].item()


def main(args):
    # Set the random number seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if not args.debug:
        # Initialize multi-processing
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = int(os.environ["LOCAL_RANK"]), torch.device(int(os.environ["LOCAL_RANK"]))
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)
    else:
        rank = 0
        world_size = 1
        device_id, device = rank, torch.device(rank+2)

    # Create directories
    if not args.debug:
        log_dir, run_dir, saved_models_dir, config_file, defaults_config_file = create_run_directories(args, rank)
    else:
        config_file = os.path.join(args.project_root_dir, "experiments", "config", args.config)
        if args.defaults_config is not None:
            defaults_config_file = os.path.join(args.project_root_dir, "experiments", "config", "defaults",
                                                args.defaults_config)
        else:
            defaults_config_file = None

    # Load configuration
    config = make_config(args, config_file, defaults_config_file)

    # Initialize logging only for rank 0
    config_dict = {s: dict(config.items(s)) for s in config.sections()}
    if not args.debug:
        if rank == 0:
            if args.use_wandb:
                wandb_summary = wandb.init(project="po_bev_unsupervised", entity="bev-projects", dir=run_dir,
                                           name="skyeye_fv_train_{}".format(args.run_name), job_type=args.mode,
                                           notes=args.comment,
                                           config=config_dict)
            else:
                wandb_summary = None
            logging.init(log_dir, "train" if args.mode == 'train' else "test")
            summary = tensorboard.SummaryWriter(log_dir)
        else:
            summary = None
            wandb_summary = None
    else:
        wandb_summary = None
        summary = tensorboard.SummaryWriter()

    # Print values for future reference
    if rank == 0:
        logging.log_info("Loading config file from: {}".format(config_file), debug=args.debug)
        config_string = json.dumps(config_dict, indent=4)
        logging.log_info(config_string, debug=args.debug)

    # Create dataloaders
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(args, config,
                       fv_num_thing=train_dataloader.dataset.fv_num_thing,
                       fv_num_stuff=train_dataloader.dataset.fv_num_stuff,
                       bev_num_thing=train_dataloader.dataset.bev_num_thing,
                       bev_num_stuff=train_dataloader.dataset.bev_num_stuff)

    # Freeze modules based on the argument inputs
    model = freeze_modules(args, model)

    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        logging.log_info("Loading snapshot from %s", args.resume, debug=args.debug)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "voxel_grid", "fv_sem_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        logging.log_info("Loading pre-trained model from %s", args.pre_train, debug=args.debug)
        pre_train_from_snapshots(model, args.pre_train, ["body", "voxel_grid", "fv_sem_head"], rank)

    # Init GPU stuff
    torch.multiprocessing.set_sharing_strategy('file_system')
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    if not args.debug:
        torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert all instances of batch norm to SyncBatchNorm
        model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    else:
        model = model.cuda(device)

    # Create optimizer
    optimizer, scheduler, batch_update, total_epochs = make_optimizer(config, model, len(train_dataloader))
    if args.resume:
        optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])
        grad_scaler.load_state_dict(snapshot['state_dict']['grad_scaler'])

    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    train_meters = {
        "loss": AverageMeter((), momentum),
        "fv_sem_loss": AverageMeter((), momentum),
        "fv_sem_conf": ConfusionMatrixMeter(train_dataloader.dataset.fv_num_categories, momentum),
    }

    if args.resume:
        starting_epoch = snapshot["training_meta"]["epoch"] + 1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]
        for name, meter in train_meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        starting_epoch = 0
        best_score = 0
        global_step = 0

    for epoch in range(starting_epoch, total_epochs):
        logging.log_info("Starting epoch %d", epoch + 1, debug=args.debug)
        if not batch_update:
            scheduler.step(epoch)

        # Run training epoch
        global_step = train(model, optimizer, grad_scaler, scheduler, train_dataloader, train_meters,
                            batch_update=batch_update, epoch=epoch, summary=summary, wandb_summary=wandb_summary,
                            device=device, log_interval=config["general"].getint("log_interval"),
                            num_epochs=total_epochs, global_step=global_step,
                            loss_weights=config['optimizer'].getstruct("loss_weights"),
                            fvsem_window_size=config['dataloader'].getint("fvsem_window_size"),
                            fvsem_step_size=config['dataloader'].getint("fvsem_step_size"),
                            debug=args.debug)

        # Save snapshot (only on rank 0)
        if not args.debug and rank == 0:
            snapshot_file = path.join(saved_models_dir, "model_latest.pth")
            logging.log_info("Saving snapshot to %s", snapshot_file)
            meters_out_dict = {k + "_meter": v.state_dict() for k, v in train_meters.items()}
            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                          body=model.module.body.state_dict(),
                          voxel_grid=model.module.voxel_grid.state_dict(),
                          fv_sem_head=model.module.fv_sem_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          grad_scaler=grad_scaler.state_dict(),
                          torch_rng=torch.get_rng_state(),
                          numpy_rng=np.random.get_state(),
                          **meters_out_dict)

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            logging.log_info("Validating epoch %d", epoch + 1, debug=args.debug)
            score = validate(model, val_dataloader, device=device, summary=summary, wandb_summary=wandb_summary,
                             global_step=global_step, epoch=epoch, num_epochs=total_epochs,
                             log_interval=config["general"].getint("log_interval"),
                             loss_weights=config['optimizer'].getstruct("loss_weights"),
                             fvsem_window_size=config['dataloader'].getint("fvsem_window_size"),
                             fvsem_step_size=config['dataloader'].getint("fvsem_step_size"),
                             img_scale=config['dataloader'].getfloat('scale'),
                             rgb_mean=config['dataloader'].getstruct('rgb_mean'),
                             rgb_std=config['dataloader'].getstruct('rgb_std'),
                             debug=args.debug)

            # Update the score on the last saved snapshot
            if not args.debug and rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot

                if score > best_score:
                    best_score = score
                    if rank == 0:
                        shutil.copy(snapshot_file, path.join(saved_models_dir, "model_best.pth"))

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            torch.cuda.empty_cache()

    logging.log_info("End of training script!", debug=args.debug)

    if not args.debug and rank == 0:
        wandb_summary.finish()

    sys.exit()


if __name__ == "__main__":
    main(parser.parse_args())
