import matplotlib as mpl
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mppatches
from skyeye.utils.sequence import pad_packed_images
from skyeye.utils.kitti_labels_merged_bev import labels as kitti_labels_bev
from skyeye.utils.kitti_labels_merged_front import labels as kitti_labels_fv
from skyeye.utils.logging import accumulate_wandb_images

def generate_visualisations(sample, results, idxs, wandb_vis_dict, **varargs):
    vis_dict = {}

    # Generate the semantic mask
    if "fv_sem_pred" in results.keys():
        if varargs['dataset'] == "Kitti360" or varargs['dataset'] == "Waymo":
            fv_sem_gt_list = make_semantic_gt_list(sample['fv_msk'], sample['fv_cat'])
        elif varargs['dataset'] == "nuScenes":
            fv_sem_gt_list = make_semantic_gt_list(sample['fv_msk'])
        vis_dict['fv_sem'] = visualise_fv_semantic(sample['img'], fv_sem_gt_list, results['fv_sem_pred'],
                                                fvsem_window_size=varargs['fvsem_window_size'],
                                                fvsem_step_size=varargs['fvsem_step_size'],
                                                scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                dataset=varargs['dataset'], rgb_mean=varargs['rgb_mean'],
                                                rgb_std=varargs['rgb_std'])

    if "bev_sem_pred" in results.keys():
        bev_sem_gt_list = make_semantic_gt_list(sample['bev_msk'], sample['bev_cat'])
        bev_combined_supervision = results['bev_combined_supervision'] if 'bev_combined_supervision' in results.keys() else None
        vis_dict['bev_sem'] = visualise_bev_semantic(sample['img'], bev_sem_gt_list, results['bev_sem_pred'],
                                                 bev_combined_supervision,
                                                 fvsem_window_size=varargs['fvsem_window_size'],
                                                 scale=0.25 / varargs['img_scale'], num_stuff=varargs['bev_num_stuff'],
                                                 dataset=varargs['dataset'], rgb_mean=varargs['rgb_mean'],
                                                 rgb_std=varargs['rgb_std'])

    if "fv_rgb_pred" in results.keys():
        vis_dict['fv_rgb_pred'] = visualise_rgb_reconsruction(sample['img'], results['fv_rgb_pred'],
                                                              fvsem_window_size=varargs['fvsem_window_size'],
                                                              fvsem_step_size=varargs['fvsem_step_size'],
                                                              scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                              rgb_mean=varargs['rgb_mean'],
                                                              rgb_std=varargs['rgb_std'])


    if "fv_depth_pred" in results.keys():
        vis_dict['fv_depth_pred'] = visualise_depth_pred(sample['img'], results['fv_depth_pred'],
                                                         fvsem_window_size=varargs['fvsem_window_size'],
                                                         scale=0.25 / varargs['img_scale'], num_stuff=varargs['fv_num_stuff'],
                                                         dataset=varargs['dataset'],
                                                         rgb_mean=varargs['rgb_mean'],
                                                         rgb_std=varargs['rgb_std'])

    if "bev_ht_pred" in results.keys():
        bev_sem_gt_list = make_semantic_gt_list(sample['bev_msk'], sample['bev_cat'])
        vis_dict['bev_ht'] = visualise_bev_heights(bev_sem_gt_list, results['bev_ht_pred'],
                                                   fvsem_window_size=varargs['fvsem_window_size'],
                                                   scale=0.25 / varargs['img_scale'], dataset=varargs['dataset'],
                                                   min_ht=1.3, max_ht=1.8)



    # Accumulate the images
    dataset_labels = get_labels(varargs['dataset'], bev=False)
    fv_sem_class_labels = {label.trainId: label.name for label in dataset_labels if label.trainId >= 0 and label.trainId != 255}
    wandb_vis_dict = accumulate_wandb_images("fv_sem", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'],
                                                 fv_sem_class_labels=fv_sem_class_labels)
    wandb_vis_dict = accumulate_wandb_images("fv_rgb_pred", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("fv_depth_pred", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("bev_sem", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])
    wandb_vis_dict = accumulate_wandb_images("bev_ht", vis_dict, wandb_vis_dict, idxs, varargs['max_vis_count'])

    return wandb_vis_dict


def visualise_fv_semantic(img, sem_gt, sem_pred, scale=0.5, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']
    fvsem_step_size = varargs['fvsem_step_size']

    if varargs['dataset'] == "Kitti360":
        idx_curr = len(sem_gt) // 2
    elif varargs['dataset'] == "Waymo" or varargs['dataset'] == "nuScenes":
        idx_curr = 0

    for b in range(len(sem_gt[0])):
        # Appending all the images next to each other
        img_b = []
        sem_pred_b = []
        sem_gt_b = []

        for ts_idx, ts in enumerate(range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size)):
            img_i = pad_packed_images(img[ts])[0][b]
            sem_pred_i = pad_packed_images(sem_pred[ts_idx])[0][b]
            sem_gt_i =  sem_gt[ts][b]

            # Scale the images based on the scale
            img_i_scaled = F.interpolate(img_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
            sem_pred_i_scaled = F.interpolate(sem_pred_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
            sem_gt_i_scaled = F.interpolate(sem_gt_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)

            # Restore the RGB image and generate the RGB images for the semantic mask
            img_i_scaled = (recover_image(img_i_scaled, varargs["rgb_mean"], varargs["rgb_std"]) * 255).type(torch.int)
            # sem_pred_i_scaled = visualise_semantic_mask_train_id(sem_pred_i_scaled, varargs['dataset'], bev=False)
            # sem_gt_i_scaled = visualise_semantic_mask_train_id(sem_gt_i_scaled, varargs['dataset'], bev=False)

            # Append all the RGB images and masks next to each other
            img_b.append(img_i_scaled)
            sem_pred_b.append(sem_pred_i_scaled)
            sem_gt_b.append(sem_gt_i_scaled)

        # Generate a long appended image
        img_b = torch.cat(img_b, dim=2)
        sem_pred_b = torch.cat(sem_pred_b, dim=2)
        sem_gt_b = torch.cat(sem_gt_b, dim=2)

        vis_dict_b = {"img": img_b, "sem_pred": sem_pred_b, "sem_gt": sem_gt_b}
        vis_list.append(vis_dict_b)

    return vis_list

def visualise_bev_semantic(img, sem_gt, sem_pred, supervision, scale=0.5, frontal_view=False, **varargs):
    vis_list = []
    fvsem_window_size = varargs['fvsem_window_size']

    if varargs['dataset'] == "Kitti360":
        idx_curr = len(sem_gt) // 2
    elif varargs['dataset'] == "Waymo" or varargs['dataset'] == "nuScenes":
        idx_curr = 0

    for b in range(len(sem_gt[0])):
        ts = idx_curr

        # This is for the BEV
        vis_b = []
        img_i = pad_packed_images(img[ts])[0][b]
        sem_pred_i = pad_packed_images(sem_pred[0])[0][b]
        sem_gt_i = sem_gt[ts][b]
        if supervision is not None:
            supervision_i = supervision[b]

        # Rotate the prediction to align it with the GT
        # sem_pred_i = torch.rot90(sem_pred_i, k=3, dims=[0, 1])
        # if supervision is not None:
        #     supervision_i = torch.rot90(supervision_i, k=1, dims=[0, 1])

        # Scale the images based on the scale
        img_i_scaled = F.interpolate(img_i.unsqueeze(0), scale_factor=scale, mode="bilinear", recompute_scale_factor=True).squeeze(0)
        sem_pred_i_scaled = F.interpolate(sem_pred_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
        sem_gt_i_scaled = F.interpolate(sem_gt_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)
        if supervision is not None:
            supervision_i_scaled = F.interpolate(supervision_i.unsqueeze(0).unsqueeze(0).type(torch.float), scale_factor=scale, mode="nearest", recompute_scale_factor=True).type(torch.int).squeeze(0)

        # Get the masked BEV prediction
        sem_pred_masked_i_scaled = sem_pred_i_scaled.clone()
        sem_pred_masked_i_scaled[sem_gt_i_scaled == 255] = 255

        # Restore the RGB image
        img_i_scaled = (recover_image(img_i_scaled, varargs['rgb_mean'], varargs['rgb_std']) * 255).type(torch.int)
        # Rotate the prediction by 90 degrees to align it with the GT
        sem_pred_i_scaled = visualise_semantic_mask_train_id(sem_pred_i_scaled, varargs['dataset'], bev=True)
        sem_gt_i_scaled = visualise_semantic_mask_train_id(sem_gt_i_scaled, varargs['dataset'], bev=True)
        sem_pred_masked_i_scaled = visualise_semantic_mask_train_id(sem_pred_masked_i_scaled, varargs['dataset'], bev=True)
        if supervision is not None:
            supervision_i_scaled = visualise_semantic_mask_train_id(supervision_i_scaled, varargs['dataset'], bev=True)

        # Align the images properly
        if varargs['dataset'] == "Kitti360":
            vis_row1 = torch.cat([sem_pred_i_scaled, sem_gt_i_scaled], dim=2)
            if supervision is not None:
                vis_row2 = torch.cat([sem_pred_masked_i_scaled, supervision_i_scaled], dim=2)

            if supervision is not None:
                vis_ts = torch.cat([img_i_scaled, vis_row1, vis_row2], dim=1)
            else:
                vis_ts = torch.cat([img_i_scaled, vis_row1], dim=1)
            vis_b.append(vis_ts)

        elif varargs['dataset'] == "Waymo" or varargs['dataset'] == "nuScenes":
            vis_ts = torch.cat([img_i_scaled, sem_pred_i_scaled, sem_gt_i_scaled], dim=1)
            vis_b.append(vis_ts)

        vis_b = torch.cat(vis_b, dim=2)
        vis_dict_b = {'bev_sem': vis_b}
        vis_list.append(vis_dict_b)

    return vis_list

def recover_image(img, rgb_mean, rgb_std):
    img = img * img.new(rgb_std).view(-1, 1, 1)
    img = img + img.new(rgb_mean).view(-1, 1, 1)
    return img


def get_labels(dataset, bev=False):
    if bev:
        if dataset == "Kitti360":
            return kitti_labels_bev
    else:
        if dataset == "Kitti360":
            return kitti_labels_fv

def visualise_semantic_mask_train_id(sem_mask, dataset, bev=False):
    dataset_labels = get_labels(dataset, bev)
    STUFF_COLOURS_TRAINID = {label.trainId: label.color for label in dataset_labels}

    sem_vis = torch.zeros((3, sem_mask.shape[1], sem_mask.shape[2]), dtype=torch.int32).to(sem_mask.device)

    # Colour the stuff
    classes = torch.unique(sem_mask)
    for stuff_label in classes:
        sem_vis[:, (sem_mask == stuff_label).squeeze()] = torch.tensor(STUFF_COLOURS_TRAINID[stuff_label.item()],
                                                                       dtype=torch.int, device=sem_mask.device).unsqueeze(1)

    return sem_vis

def plot_confusion_matrix(conf_mat, num_classes, dataset, bev=False):
    labels = get_labels(dataset, bev=bev)
    ignore_classes = [255, -1]

    # Get the class names
    seen_ids = []
    class_labels = []
    for l in labels:
        if (l.trainId in seen_ids) or (l.trainId in ignore_classes):
            continue
        seen_ids.append(l.trainId)
        class_labels.append(l.name)

    # Get the important part of the confusion matrix
    conf_mat_np = conf_mat[:num_classes, :num_classes]

    # Get the ratio. Row elts + Col elts - Diagonal elt (it is computed twice)
    conf_mat_np = conf_mat_np / ((conf_mat_np.sum(dim=0) + conf_mat_np.sum(dim=1) - conf_mat_np.diag()) + 1e-8)  # Small number added to avoid nan
    conf_mat_np = conf_mat_np.cpu().detach().numpy()

    # Plot the confusion matrix
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    conf_mat_plt = sns.heatmap(conf_mat_np * 100, annot=True, fmt=".2g", vmin=0.0, vmax=100., square=True,
                               xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 7}, ax=ax)

    return conf_mat_plt

def save_semantic_output(sample, sample_category, save_tuple, bev=False, woskyveg=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = [os.path.join(save_dir_rgb, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]
    img_name = [os.path.join(save_dir, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]

    # Generate the numpy image and save the image using OpenCV
    for idx, (sample_ts, img_name_ts, img_name_rgb_ts) in enumerate(zip(sample, img_name, img_name_rgb)):
        # Save the raw version of the mask
        sample_ts_orig = sample_ts.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if bev:
            sample_ts_orig = cv2.rotate(sample_ts_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_ts, sample_ts_orig)

        # Save the RGB version of the mask
        sample_ts_rgb = visualise_semantic_mask_train_id(sample_ts, varargs['dataset'], bev=bev)
        sample_ts_rgb = sample_ts_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_rgb = cv2.cvtColor(sample_ts_rgb, cv2.COLOR_RGB2BGR)
        if bev:
            sample_ts_rgb = cv2.rotate(sample_ts_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_rgb_ts, sample_ts_rgb)


def save_semantic_output_with_rgb_overlay(sample, rgb, sample_category, save_tuple, bev=False, woskyveg=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        img_save_dir = os.path.join(save_path, "img_rgb")
        stacked_save_dir = os.path.join(save_path, "stacked_pred_rgb")
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        img_save_dir = os.path.join(save_path, "img_rgb")
        stacked_save_dir = os.path.join(save_path, "stacked_pred_rgb")

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(stacked_save_dir):
        os.makedirs(stacked_save_dir)

    # Generate the numpy image and save the image using OpenCV
    stacked_images = []
    for idx, (sample_ts, rgb_ts) in enumerate(zip(sample, rgb)):
        # Save the RGB version of the mask
        sample_ts_rgb = visualise_semantic_mask_train_id(sample_ts, varargs['dataset'], bev=bev)
        sample_ts_rgb = sample_ts_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_rgb = cv2.cvtColor(sample_ts_rgb, cv2.COLOR_RGB2BGR)

        rgb_ts_recover = recover_image(rgb_ts, varargs['rgb_mean'], varargs["rgb_std"]).squeeze(0) * 255
        rgb_ts_recover = rgb_ts_recover.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rgb_ts_recover = cv2.cvtColor(rgb_ts_recover, cv2.COLOR_RGB2BGR)

        # Save the RGB image for the first timestep
        if idx == 0:
            img_save_name = os.path.join(img_save_dir, "{}.png".format(sample_name[0]))
            cv2.imwrite(img_save_name, rgb_ts_recover)
            stacked_images.append(cv2.resize(rgb_ts_recover, dsize=(0, 0), fx=0.25, fy=0.25))

        # Blend RGB and FV Pred
        sample_ts_overlay = cv2.addWeighted(sample_ts_rgb, 0.6, rgb_ts_recover, 0.4, 0.0)

        overlay_save_name = os.path.join(save_dir_rgb, "{}_{}.png".format(sample_name[0], idx))
        cv2.imwrite(overlay_save_name, sample_ts_overlay)
        stacked_images.append(cv2.resize(sample_ts_overlay, dsize=(0, 0), fx=0.25, fy=0.25))

    stacked_img = np.concatenate(stacked_images, axis=1)
    stacked_save_name = os.path.join(stacked_save_dir, "{}.png".format(sample_name[0]))
    cv2.imwrite(stacked_save_name, stacked_img)


def save_semantic_masked_output(sample_pred, sample_gt, sample_category, save_tuple, bev=False, **varargs):
    if save_tuple is None:
        return

    save_path, sample_name = save_tuple[0], save_tuple[1]

    # Check if the directory exists. If not create it
    cam_name = varargs['cam_name'] if "cam_name" in varargs.keys() else None
    if cam_name is not None:
        save_dir_rgb = os.path.join(save_path, cam_name, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, cam_name, sample_category)
    else:
        save_dir_rgb = os.path.join(save_path, "{}_rgb".format(sample_category))
        save_dir = os.path.join(save_path, sample_category)

    if not os.path.exists(save_dir_rgb):
        os.makedirs(save_dir_rgb)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_name_rgb = [os.path.join(save_dir_rgb, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]
    img_name = [os.path.join(save_dir, "{}.png".format(sample_name_i)) for sample_name_i in sample_name]

    # Generate the numpy image and save the image using OpenCV
    for idx, (sample_pred_ts, sample_gt_ts, img_name_ts, img_name_rgb_ts) in enumerate(zip(sample_pred, sample_gt, img_name, img_name_rgb)):
        # Mask the prediction using the GT
        sample_pred_ts[sample_gt_ts == 255] = 255

        # Save the raw version of the mask
        sample_ts_orig_masked = sample_pred_ts.permute(1, 2, 0).cpu().numpy().astype(np.uint16)
        sample_ts_orig_masked = cv2.rotate(sample_ts_orig_masked, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_ts, sample_ts_orig_masked)

        # Save the RGB version of the mask
        sample_ts_masked_rgb = visualise_semantic_mask_train_id(sample_pred_ts, varargs['dataset'], bev=bev)
        sample_ts_masked_rgb = sample_ts_masked_rgb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        sample_ts_masked_rgb = cv2.cvtColor(sample_ts_masked_rgb, cv2.COLOR_RGB2BGR)
        sample_ts_masked_rgb = cv2.rotate(sample_ts_masked_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(img_name_rgb_ts, sample_ts_masked_rgb)


def make_semantic_gt_list(msk, cat=None):
    sem_out = []
    if cat is not None:
        for msk_timestep, cat_timestep in zip(msk, cat):
            sem_timestep_out = []
            for msk_i, cat_i in zip(msk_timestep, cat_timestep):
                msk_i = msk_i.squeeze(0)
                sem_timestep_out.append(cat_i[msk_i])
            sem_out.append(sem_timestep_out)
    else:
        for msk_timestep in msk:
            sem_timestep_out = []
            for msk_i in msk_timestep:
                msk_i = msk_i.squeeze(0)
                sem_timestep_out.append(msk_i)
            sem_out.append(sem_timestep_out)
    return sem_out