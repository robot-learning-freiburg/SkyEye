import copy
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skyeye.utils.sequence import pad_packed_images
from skyeye.utils.visualisation import save_semantic_output, save_semantic_masked_output, save_semantic_output_with_rgb_overlay

NETWORK_INPUTS_FV = ["img", "fv_msk", "fv_cat", "fv_iscrowd", "fv_intrinsics", "ego_pose"]
NETWORK_INPUTS_BEV = ["img", "bev_msk", "bev_cat", "bev_iscrowd", "bev_plabel", "fv_intrinsics", "ego_pose"]

class UnsupervisedBevNet(nn.Module):
    def __init__(self,
                 body,
                 voxel_grid,
                 fv_sem_head,
                 bev_sem_head,
                 voxel_grid_algo,
                 fv_sem_algo,
                 bev_sem_algo,
                 dataset,
                 fv_classes=None,
                 bev_classes=None,
                 fv_sky_index=None,
                 fv_veg_index=None):
        super(UnsupervisedBevNet, self).__init__()

        # Backbone
        self.body = body

        # Transformer
        self.voxel_grid = voxel_grid

        # Modules
        self.fv_sem_head = fv_sem_head
        self.bev_sem_head = bev_sem_head

        # Algorithms
        self.voxel_grid_algo = voxel_grid_algo
        self.fv_sem_algo = fv_sem_algo
        self.bev_sem_algo = bev_sem_algo

        # Params
        self.dataset = dataset
        self.fv_num_classes = fv_classes['total']
        self.fv_num_stuff = fv_classes["stuff"]
        self.bev_num_classes = bev_classes['total']
        self.bev_num_stuff = bev_classes['stuff']
        self.fv_sky_index = fv_sky_index
        self.fv_veg_index = fv_veg_index

    def _prepare_inputs(self, msk, cat, iscrowd, front=True):
        if front:
            num_stuff = self.fv_num_stuff
        else:
            num_stuff = self.bev_num_stuff

        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, sem_wo_sky_out, po_out, po_vis_out = [], [], [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i in zip(msk, cat, iscrowd):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i > 0)

            if valid.any().item():
                cat_out.append(cat_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i > 0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_msk_i = cat_i[msk_i]
            sem_out.append(sem_msk_i)

            # Get the FV image in terms of the BEV labels. This basically eliminates sky in the FV image
            if front:
                sem_wo_sky_veg_i = copy.deepcopy(sem_msk_i)
                sem_wo_sky_veg_i[sem_wo_sky_veg_i == self.fv_sky_index] = 255
                sem_wo_sky_veg_i[sem_wo_sky_veg_i == self.fv_veg_index] = 255
                for lbl in torch.unique(sem_wo_sky_veg_i):
                    decr_ctr = 0
                    if (lbl > self.fv_sky_index) and (lbl != 255):
                        decr_ctr += 1
                    if (lbl > self.fv_veg_index) and (lbl != 255):
                        decr_ctr += 1
                    sem_wo_sky_veg_i[sem_wo_sky_veg_i == lbl] = lbl - decr_ctr
                sem_wo_sky_out.append(sem_wo_sky_veg_i)

        if front:
            return cat_out, iscrowd_out, ids_out, sem_out, sem_wo_sky_out
        else:
            return cat_out, iscrowd_out, ids_out, sem_out

    def forward(self, img, fv_msk=None, fv_cat=None, fv_iscrowd=None, bev_msk=None, bev_cat=None, bev_iscrowd=None,
                bev_plabel=None, fv_intrinsics=None, ego_pose=None, transform_status=None,
                do_loss=False, use_fv=False, use_bev=False, fvsem_window_size=None, fvsem_step_size=None,
                save_tuple=None, rgb_mean=None, rgb_std=None):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Process plabels to have the correct format:
        if bev_plabel is not None:
            bev_plabel, bev_plabel_valid_img_size_i = pad_packed_images(bev_plabel[0])
            bev_plabel = list(torch.tensor_split(bev_plabel, bev_plabel.shape[0], dim=0))
            bev_plabel = [elem.squeeze() for elem in bev_plabel]

        # Get the index for the data at the current time step. This is exactly in the middle of the list
        idx_curr = len(img) // 2
        fv_img_shape = pad_packed_images(img[idx_curr])[0].shape[-2:]

        fv_sem_loss, fv_sem_conf_mat, fv_sem_pred, fv_sem_logits = [], [], [], []
        bev_sem_loss, bev_sem_conf_mat, bev_sem_pred, bev_sem_logits = [], [], [], []

        # ***** FV + BEV SEGMENTATION *****
        # Iterate through only the future frames for the FV segmentation.
        for i in range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size):
            # Get/Prepare the input data and ground truth labels
            img_i, _ = pad_packed_images(img[i])
            ego_pose_i = ego_pose[i]
            fv_intrinsics_i = fv_intrinsics[i]

            if (fv_msk is not None) and (fv_msk[i] is not None):
                fv_msk_i, fv_valid_size_i = pad_packed_images(fv_msk[i])
                fv_img_size_i = fv_msk_i.shape[-2:]
                if self.dataset == "Kitti360":
                    fv_cat_i = fv_cat[i]
                    fv_iscrowd_i = fv_iscrowd[i]

            if (bev_msk is not None) and (bev_msk[i] is not None):
                bev_msk_i, bev_valid_size_i = pad_packed_images(bev_msk[i])
                bev_img_size_i = bev_msk_i.shape[-2:]
                bev_cat_i = bev_cat[i]
                bev_iscrowd_i = bev_iscrowd[i]

            # Prepare the input data and the groundtruth labels
            if fv_msk is not None:
                fv_cat_i, fv_iscrowd_i, fv_ids_i, fv_sem_gt_i, fv_sem_wo_sky_gt_i = self._prepare_inputs(fv_msk_i, fv_cat_i, fv_iscrowd_i, front=True)
            if bev_msk is not None:
                bev_cat_i, bev_iscrowd_i, bev_ids_i, bev_sem_gt_i = self._prepare_inputs(bev_msk_i, bev_cat_i, bev_iscrowd_i, front=False)

            # Generate the voxel grid for all the frames
            ms_feat_i = self.body(img_i)
            feat_voxel_i, feat_merged_2d_i, implicit_depth_dist_i, implicit_depth_dist_unproj_i, vxl_to_fv_idx_map_i = self.voxel_grid(ms_feat_i, fv_intrinsics_i, fv_img_shape=fv_img_shape)
            del ms_feat_i

            if i == idx_curr:
                feat_voxel_curr = feat_voxel_i

            ############################# ONLY VOXEL WARPING ####################################
            if do_loss and i != idx_curr:
                feat_voxel_i_warped = self.voxel_grid_algo.ego_gt_warper(feat_voxel_curr, ego_pose[idx_curr], ego_pose_i, transform_status)
            else:
                feat_voxel_i_warped = feat_voxel_curr

            ############################# FV SEGMENTATION ##############################
            if use_fv:
                # Orthographic to perspective distortion
                feat_voxel_i_warped_persp = self.voxel_grid_algo.apply_perspective_distortion(feat_voxel_i_warped, fv_img_size_i, fv_intrinsics_i)

                if do_loss:
                    fv_sem_loss_i, fv_sem_conf_mat_i, fv_sem_pred_i, fv_sem_logits_i = self.fv_sem_algo.training_fv(self.fv_sem_head,
                                                                                                                    feat_voxel_i_warped_persp,
                                                                                                                    fv_sem_gt_i, fv_valid_size_i,
                                                                                                                    fv_img_size_i)
                else:
                    fv_sem_pred_i, fv_sem_logits_i = self.fv_sem_algo.inference_fv(self.fv_sem_head,
                                                                                   feat_voxel_i_warped_persp,
                                                                                   fv_valid_size_i,
                                                                                   fv_intrinsics_i)
                    fv_sem_loss_i, fv_sem_conf_mat_i = None, None

                fv_sem_loss.append(fv_sem_loss_i)
                fv_sem_conf_mat.append(fv_sem_conf_mat_i)
                fv_sem_pred.append(fv_sem_pred_i)
                fv_sem_logits.append(fv_sem_logits_i)
            else:
                fv_sem_loss = None
                fv_sem_conf_mat = None
                fv_sem_pred = None
                fv_sem_logits = None

            if use_bev:
                # Generate the BEV for the current frame only.
                if i == idx_curr:
                    if do_loss:  # During training
                        bev_sem_loss_i, bev_sem_pred_i, bev_sem_logits_i, bev_height_map_i = self.bev_sem_algo.training_bev(self.bev_sem_head, feat_voxel_i, bev_plabel)
                        bev_sem_conf_i = self.bev_sem_algo.compute_bev_metrics_with_gt(bev_sem_pred_i, bev_sem_gt_i)
                    else:
                        bev_sem_pred_i , bev_sem_logits_i, bev_height_map_i = self.bev_sem_algo.inference_bev(self.bev_sem_head, feat_voxel_i)
                        bev_sem_loss_i, bev_sem_conf_i = None, None

                    bev_sem_loss.append(bev_sem_loss_i)
                    bev_sem_pred.append(bev_sem_pred_i)
                    bev_sem_logits.append(bev_sem_logits_i)
                    bev_sem_conf_mat.append(bev_sem_conf_i)
            else:
                bev_sem_loss = None
                bev_sem_pred = None
                bev_sem_conf = None
                bev_sem_logits = None

        if use_fv and (fv_sem_loss is not None) and len(fv_sem_loss) > 0:
            fv_sem_loss_count = len(fv_sem_loss)
            fv_sem_loss_weights = torch.linspace(1, 0.2, fv_sem_loss_count).tolist()
            fv_sem_loss_sum = sum([w * l for w, l in zip(fv_sem_loss_weights, fv_sem_loss)])
            fv_sem_loss_net = fv_sem_loss_sum / fv_sem_loss_count
        else:
            fv_sem_loss_net = torch.tensor(0.).to(img[idx_curr].device)

        if use_bev and (bev_sem_loss is not None) and len(bev_sem_loss) > 0:
            bev_sem_loss_net = sum(bev_sem_loss) / len(bev_sem_loss)
        else:
            bev_sem_loss_net = torch.tensor(0.).to(img[idx_curr].device)

        if use_fv and (fv_msk is not None):
            fv_sem_conf_mat_net = torch.zeros(self.fv_num_classes, self.fv_num_classes, dtype=torch.double).to(fv_sem_conf_mat[0].device)
            for conf_mat in fv_sem_conf_mat:
                fv_sem_conf_mat_net += conf_mat

        if use_bev and (self.bev_sem_algo is not None):
            bev_sem_conf_mat_net = torch.zeros(self.bev_num_classes, self.bev_num_classes, dtype=torch.double).to(bev_sem_conf_mat[0].device)
            for conf_mat in bev_sem_conf_mat:
                bev_sem_conf_mat_net += conf_mat

        # LOSSES
        if use_fv:
            loss["fv_sem_loss"] = fv_sem_loss_net
        if use_bev:
            loss['bev_sem_loss'] = bev_sem_loss_net

        # RESULTS
        if use_fv:
            result["fv_sem_pred"] = fv_sem_pred
            result['fv_sem_logits'] = fv_sem_logits
        if use_bev:
            result["bev_sem_pred"] = bev_sem_pred
            result['bev_sem_logits'] = bev_sem_logits

        # STATS
        if do_loss:
            if use_fv:
                stats['fv_sem_conf'] = fv_sem_conf_mat_net
            if use_bev:
                stats['bev_sem_conf'] = bev_sem_conf_mat_net


        # Save all the required outputs here
        if save_tuple is not None:
            if bev_sem_pred is not None:
                bev_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in bev_sem_pred]
                save_semantic_output(bev_sem_pred_unpack, "bev_sem_pred", save_tuple, bev=True, dataset=self.dataset)
            if bev_msk is not None:
                bev_sem_gt_unpack = [pad_packed_images(gt)[0] for gt in bev_msk]
                bev_sem_gt_unpack = [self._prepare_inputs(bev_sem_gt_unpack[vis_ts], bev_cat[vis_ts], bev_iscrowd[vis_ts], front=False)[-1][0] for vis_ts in range(len(bev_sem_gt_unpack))]
                bev_sem_gt_unpack = [gt.unsqueeze(0) for gt in bev_sem_gt_unpack]
                save_semantic_output(bev_sem_gt_unpack, "bev_sem_gt", save_tuple, bev=True, dataset=self.dataset)
                if bev_sem_pred is not None:
                    bev_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in bev_sem_pred]
                    save_semantic_masked_output(bev_sem_pred_unpack, bev_sem_gt_unpack, "bev_sem_pred_masked", save_tuple, bev=True, dataset=self.dataset)
            if fv_sem_pred is not None:
                fv_sem_pred_unpack = [pad_packed_images(pred)[0] for pred in fv_sem_pred]
                save_semantic_output(fv_sem_pred_unpack, "fv_sem_pred", save_tuple, bev=False, dataset=self.dataset)

                img_unpack = [pad_packed_images(rgb)[0] for rgb in img]
                img_unpack = [img_unpack[i] for i in range(idx_curr, min(len(img), idx_curr + fvsem_window_size + 1), fvsem_step_size)]
                save_semantic_output_with_rgb_overlay(fv_sem_pred_unpack, img_unpack, "fv_sem_pred_rgb_overlay",
                                                      save_tuple, bev=False, dataset=self.dataset,
                                                      rgb_mean=rgb_mean, rgb_std=rgb_std)
            if fv_msk is not None:
                # Without vegetation and sky
                fv_sem_gt_unpack = [pad_packed_images(gt)[0] for gt in fv_msk]
                if self.dataset == "Kitti360":
                    fv_sem_gt_unpack_woskyveg = [
                        self._prepare_inputs(fv_sem_gt_unpack[vis_ts], fv_cat[vis_ts], fv_iscrowd[vis_ts], front=True)[-1][
                            0] for vis_ts in range(len(fv_sem_gt_unpack))]
                    fv_sem_gt_unpack_woskyveg = [gt.unsqueeze(0) for gt in fv_sem_gt_unpack_woskyveg]
                    save_semantic_output(fv_sem_gt_unpack_woskyveg, "fv_sem_woskyveg_gt", save_tuple, bev=False,
                                         woskyveg=True, dataset=self.dataset)

                # With all classes
                if self.dataset == "Kitti360":
                    fv_sem_gt_unpack = [self._prepare_inputs(fv_sem_gt_unpack[vis_ts], fv_cat[vis_ts], fv_iscrowd[vis_ts], front=True)[-2][0] for vis_ts in range(len(fv_sem_gt_unpack))]
                    fv_sem_gt_unpack = [gt.unsqueeze(0) for gt in fv_sem_gt_unpack]
                elif self.dataset == "nuScenes":
                    fv_sem_gt_unpack = [fv_msk_i.squeeze(1) for fv_msk_i in fv_sem_gt_unpack]
                save_semantic_output(fv_sem_gt_unpack, "fv_sem_gt", save_tuple, bev=False, dataset=self.dataset)

        return loss, result, stats
