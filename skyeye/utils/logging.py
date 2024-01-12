import logging
from math import log10
from os import path
from torch import distributed
from collections import OrderedDict
import wandb

from .meters import AverageMeter, ConstantMeter

_NAME = "SKYEYE"

def  _current_total_formatter(current, total):
    return "[{}/{}]".format(current, total)


def init(log_dir, name, debug=True):
    logger = logging.getLogger(_NAME)
    logger.setLevel(logging.DEBUG)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Setup file logging
    if not debug:
        file_handler = logging.FileHandler(path.join(log_dir, name + ".log"), mode="w")
        file_formatter = logging.Formatter(fmt="%(levelname).1s %(asctime)s %(message)s", datefmt="%y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

def get_logger():
    return logging.getLogger(_NAME)

def iteration(summary, phase, global_step, epoch, num_epochs, step, num_steps, values, multiple_lines=False, **params):
    # Build message and write summary
    msg = _current_total_formatter(epoch, num_epochs) + " " + _current_total_formatter(step, num_steps)
    for k, v in values.items():
        if isinstance(v, AverageMeter):
            msg += "\n" if multiple_lines else "" + "\t{}={:.5f} ({:.5f})".format(k, v.value.item(), v.mean.item())
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v.value.item(), global_step)
        elif isinstance(v, ConstantMeter):
            msg += "\n" if multiple_lines else "" + "\t{}={:.5f}".format(k, v.value.item())
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v.value.item(), global_step)
        else:
            msg += "\n" if multiple_lines else "" + "\t{}={:.5f}".format(k, v)
            if summary is not None:
                summary.add_scalar("{}/{}".format(phase, k), v, global_step)

    # Write log
    log_info(msg, debug=params['debug'])

def log_info(msg, *args, **kwargs):
    if "debug" in kwargs.keys():
        print(msg % args)
    else:
        if distributed.get_rank() == 0:
            get_logger().info(msg, *args, **kwargs)

def log_miou(label, miou, classes):
    logger = get_logger()
    padding = max(len(cls) for cls in classes)

    logger.info("---------------- {} ----------------".format(label))
    for miou_i, class_i in zip(miou, classes):
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(class_i, miou_i.item()))

def log_scores(label, scores):
    logger = get_logger()
    padding = max(len(cls) for cls in scores.keys())

    logger.info("---------------- {} ----------------".format(label))
    for score_label, score_value in scores.items():
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(score_label, score_value.item()))


def log_iter(mode, meters, time_meters, metrics, batch=True, **kwargs):
    assert mode in ['train', 'val', 'test'], "Mode must be either 'train', 'val', or 'test'!"
    iou = ["fv_sem_conf", "bev_sem_conf"]

    log_entries = []

    if kwargs['lr'] is not None:
        log_entries = [("lr", kwargs['lr'])]

    meters_keys = list(meters.keys())
    meters_keys.sort()
    for meter_key in meters_keys:
        if meter_key in iou:
            log_key = meter_key
            log_value = meters[meter_key].iou.mean().item()
        else:
            if not batch:
                log_value = meters[meter_key].mean.item()
            else:
                log_value = meters[meter_key]
            log_key = meter_key

        log_entries.append((log_key, log_value))

    time_meters_keys = list(time_meters.keys())
    time_meters_keys.sort()
    for meter_key in time_meters_keys:
        log_key = meter_key
        if not batch:
            log_value = time_meters[meter_key].mean.item()
        else:
            log_value = time_meters[meter_key]
        log_entries.append((log_key, log_value))

    if metrics is not None:
        metrics_keys = list(metrics.keys())
        metrics_keys.sort()
        for metric_key in metrics_keys:
            log_key = metric_key
            if not batch:
                log_value = metrics[log_key].mean.item()
            else:
                log_value = metrics[log_key]
            log_entries.append((log_key, log_value))

    iteration(kwargs["summary"], mode, kwargs["global_step"], kwargs["epoch"] + 1, kwargs["num_epochs"],
                      kwargs['curr_iter'], kwargs['num_iters'], OrderedDict(log_entries), debug=kwargs['debug'])


def log_wandb(wandb_summary, mode, meters, time_meters, metrics, batch, **kwargs):
    assert mode in ['train', 'val', 'test'], "Mode must be either 'train', 'val', or 'test'!"

    iou = ["fv_sem_conf", "bev_sem_conf"]
    suffix = "batch" if batch else "total"

    log_dict = {}
    for meter_key in meters.keys():
        log_key = "{}/{}_{}".format(mode, meter_key, suffix)
        if batch:
            if meter_key in iou:
                log_value = meters[meter_key].iou.mean().item()
                # pass
            else:
                log_value = meters[meter_key].value.item()
        else:
            if meter_key in iou:
                log_value = meters[meter_key].iou.mean().item()
                # pass
            else:
                log_value = meters[meter_key].mean.item()
        log_dict[log_key] = log_value

    for meter_key in time_meters.keys():
        log_key = "{}/{}_{}".format(mode, meter_key, suffix)
        if batch:
            log_value = time_meters[meter_key].value.item()
        else:
            log_value = time_meters[meter_key].mean.item()
        log_dict[log_key] = log_value

    if metrics is not None:
        for metric_key in metrics.keys():
            log_key = "{}_metrics/{}".format(mode, metric_key)
            if batch:
                log_value = metrics[metric_key].value.item()
            else:
                log_value = metrics[metric_key].mean.item()
            log_dict[log_key] = log_value

    # Other stuff
    log_dict['epoch'] = kwargs["epoch"]
    if kwargs['lr'] is not None:
        log_dict["{}/lr".format(mode)] = kwargs['lr']

    if not kwargs['debug']:
        wandb_summary.log(log_dict, step=kwargs["global_step"])


def make_wandb_entry(key, caption_prefix, vis_dict, wandb_vis_dict, image_names, max_vis_count, count_key, wandb_key):
    for batch_idx, dict_list in enumerate(vis_dict[key]):
        sample_name = image_names[len(image_names) // 2][batch_idx]

        if count_key not in wandb_vis_dict.keys():
            wandb_vis_dict[count_key] = 0

        if wandb_vis_dict[count_key] < max_vis_count:
            wandb_obj = wandb.Image(dict_list[key].permute(1, 2, 0).detach().cpu().numpy(),
                                    caption="{}: {}".format(caption_prefix, sample_name))

            if wandb_key not in wandb_vis_dict.keys():
                wandb_vis_dict[wandb_key] = [wandb_obj]
            else:
                wandb_vis_dict[wandb_key].append(wandb_obj)

            wandb_vis_dict[count_key] += 1
        else:
            break
    return wandb_vis_dict


def accumulate_wandb_images(key, vis_dict, wandb_vis_dict, image_names, max_vis_count, **varargs):
    if key in vis_dict.keys():
        count_key = 'count_{}'.format(key)
        wandb_key = "wandb_{}".format(key)
        if key == "fv_sem":
            class_labels = varargs['fv_sem_class_labels']
            for batch_idx, fv_sem_dict in enumerate(vis_dict[key]):
                sample_name = image_names[len(image_names)//2][batch_idx]

                if count_key not in wandb_vis_dict.keys():
                    wandb_vis_dict[count_key] = 0

                if wandb_vis_dict[count_key] < max_vis_count:
                    masks = {
                        "predictions": {
                            'mask_data': fv_sem_dict['sem_pred'].squeeze(0).detach().cpu().numpy(),
                            'class_labels': class_labels
                    },
                        "ground_truth": {
                            "mask_data": fv_sem_dict['sem_gt'].squeeze(0).detach().cpu().numpy(),
                            'class_labels': class_labels
                        }
                    }

                    wandb_obj = wandb.Image(fv_sem_dict['img'].permute(1, 2, 0).detach().cpu().numpy(),
                                            masks=masks, caption="FV Sem: {}".format(sample_name))

                    if wandb_key not in wandb_vis_dict.keys():
                        wandb_vis_dict[wandb_key] = [wandb_obj]
                    else:
                        wandb_vis_dict[wandb_key].append(wandb_obj)

                    wandb_vis_dict[count_key] += 1
                else:
                    break

        elif key == "bev_sem":
            caption_prefix = "BEV Sem"
            wandb_vis_dict = make_wandb_entry(key, caption_prefix, vis_dict, wandb_vis_dict, image_names, max_vis_count,
                                              count_key, wandb_key)

    return wandb_vis_dict