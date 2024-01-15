import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import kornia
import kornia.geometry.linalg

class VoxelGridAlgo:
    def __init__(self, feat_scale=None, resolution=None, y_extents=None):
        self.feat_scale = feat_scale
        self.y_extents = y_extents
        if resolution is not None and feat_scale is not None:
            self.resolution = resolution / feat_scale
        else:
            self.resolution = None

    @staticmethod
    def compute_relative_transformation(pose_src, pose_dst, transform_status, homogeneous=False):
        # Check if the matrices are homogeneous. If not, make them homogeneous
        if not ((pose_dst.shape[1] == 4) and (pose_dst.shape[2] == 4)):
            last_row = torch.zeros([1, 4], dtype=torch.float).to(pose_dst.device)
            last_row[0, 3] = 1
            last_row = torch.stack([last_row] * pose_dst.shape[0], dim=0)
            pose_dst = torch.cat([pose_dst, last_row], dim=1)

        if not ((pose_src.shape[1] == 4) and (pose_src.shape[2] == 4)):
            last_row = torch.zeros([1, 4], dtype=torch.float).to(pose_src.device)
            last_row[0, 3] = 1
            last_row = torch.stack([last_row] * pose_src.shape[0], dim=0)
            pose_src = torch.cat([pose_src, last_row], dim=1)

        # Convert to double to get the required precision
        pose_dst = pose_dst.type(torch.double)
        pose_src = pose_src.type(torch.double)

         # Get the transformation to transform from pose_src to pose_dst
        transform_src2dst = torch.matmul(torch.inverse(pose_dst), pose_src).type(torch.float32)

        # Flip the relative transformation if the images are flipped in the data augmentation
        for b_idx in range(transform_src2dst.shape[0]):
            if transform_status[b_idx]['flip']:
                transform_src2dst[b_idx] = VoxelGridAlgo.flip_relative_pose(transform_src2dst[b_idx])

        if homogeneous:
            return transform_src2dst
        else:
            transform_src2dst = transform_src2dst[:, :3, :]  # Get the Bx3x4 matrix back
            return transform_src2dst


    def ego_gt_warper(self, voxel_grid, ego_pose_src, ego_pose_dst, transform_status):
        """ Warp the voxel grid using the relative pose between the curr and i_th frames"""

        # Compute the relative transform between the two image frames.
        # This is the transformation to transform the curr frame to the ith frame
        transform_src2dst = self.compute_relative_transformation(ego_pose_src, ego_pose_dst, transform_status, homogeneous=False)

        # Warp the voxel grid using the relative transformation
        transform_src2dst[:, :3, 3] = transform_src2dst[:, :3, 3] / self.resolution
        voxel_grid_warped = kornia.geometry.transform.warp_affine3d(voxel_grid, transform_src2dst[:, :3], dsize=voxel_grid.shape[2:])
        return voxel_grid_warped

    @staticmethod
    def flip_relative_pose(T_rel):
        T_flip = torch.tensor([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=torch.float32, device=T_rel.device)
        trans = T_rel[:3, 3].unsqueeze(1)
        trans_flip = torch.matmul(T_flip, trans)  # Flip the translation

        # Get the rotation angles
        rot_mat = T_rel[:3, :3]
        theta_z = roll = torch.atan2(rot_mat[1, 0], rot_mat[1, 1])
        theta_x = pitch = torch.atan2(-rot_mat[1, 2], torch.sqrt(rot_mat[0, 2] ** 2 + rot_mat[2, 2] ** 2))
        theta_y = yaw = torch.atan2(rot_mat[0, 2], rot_mat[2, 2])

        # Flip the rotation angles
        pitch_flip = pitch
        yaw_flip = -yaw
        roll_flip = -roll

        R_pitch = torch.tensor([[1, 0, 0],
                                [0, torch.cos(pitch_flip), -torch.sin(pitch_flip)],
                                [0, torch.sin(pitch_flip), torch.cos(pitch_flip)]], dtype=torch.float32, device=T_rel.device)

        R_yaw = torch.tensor([[torch.cos(yaw_flip), 0, torch.sin(yaw_flip)],
                              [0, 1, 0],
                              [-torch.sin(yaw_flip), 0, torch.cos(yaw_flip)]], dtype=torch.float32, device=T_rel.device)

        R_roll = torch.tensor([[torch.cos(roll_flip), -torch.sin(roll_flip), 0],
                               [torch.sin(roll_flip), torch.cos(roll_flip), 0],
                               [0, 0, 1]], dtype=torch.float32, device=T_rel.device)

        rot_mat_flip = torch.matmul(R_yaw, torch.matmul(R_pitch, R_roll))

        T_rel_flip = torch.zeros([4, 4], dtype=torch.float32, device=T_rel.device)
        T_rel_flip[:3, :3] = rot_mat_flip
        T_rel_flip[:3, 3] = trans_flip.squeeze(1)
        T_rel_flip[3, 3] = 1

        return T_rel_flip

    def map_frustum_idxs_to_voxel_grid(self, projection_intrinsic, grid_dims, fv_img_size_i, dtype, device):
        # This follows the camera coordinate system. So X is to the right, Y is downwards, and Z goes into the scene
        B = projection_intrinsic.shape[0]
        voxel_size_x, voxel_size_y, voxel_size_z = grid_dims

        feat_shape = (fv_img_size_i[0] * self.feat_scale, fv_img_size_i[1] * self.feat_scale)

        u_coords = torch.arange(0, feat_shape[1], 1, dtype=dtype, device=device)
        v_coords = torch.arange(0, feat_shape[0], 1, dtype=dtype, device=device)
        z_coords = torch.arange(0, voxel_size_z, 1, dtype=dtype, device=device)
        coords = torch.stack([torch.stack(torch.meshgrid([u_coords, v_coords, z_coords]))] * B)

        cx = projection_intrinsic[:, 0, 2].view(projection_intrinsic.shape[0], 1, 1, 1)
        fx = projection_intrinsic[:, 0, 0].view(projection_intrinsic.shape[0], 1, 1, 1)
        cy = projection_intrinsic[:, 1, 2].view(projection_intrinsic.shape[0], 1, 1, 1)
        fy = projection_intrinsic[:, 1, 1].view(projection_intrinsic.shape[0], 1, 1, 1)

        coords[:, 0] = ((coords[:, 0] - cx) / fx) * coords[:, 2]
        coords[:, 1] = ((coords[:, 1] - cy) / fy) * coords[:, 2]

        # We need to translate the coords to the centre of the voxel grid because the px and py move it there.
        # Z is the same in both the frustum and voxel grid, so there is no need to change that.
        coords[:, 0] += voxel_size_x / 2
        # coords[:, 1] += voxel_size_y / 2
        coords[:, 1] += voxel_size_y * (abs(self.y_extents[0]) / (abs(self.y_extents[0] - self.y_extents[1])))

        # Normalise the coords to between -1 and 1 so that we can apply grid_sample
        coords[:, 0] = ((coords[:, 0] / (voxel_size_x - 1)) - 0.5) * 2
        coords[:, 1] = ((coords[:, 1] / (voxel_size_y - 1)) - 0.5) * 2
        coords[:, 2] = ((coords[:, 2] / (voxel_size_z - 1)) - 0.5) * 2

        # Permute the axes so that the dimensions are N x D x H x W x 3
        coords = coords.permute(0, 4, 3, 2, 1)

        return coords

    def apply_perspective_distortion(self, feat_voxel_i_warped, fv_img_size_i, intrinsics):
        D, H, W = feat_voxel_i_warped.shape[-3:]
        grid_dims = [W, H, D]
        intrinsics_scaled = intrinsics / 8 # The feature maps from the encoder are of scale 1/8
        intrinsics_scaled[:, 2, 2] = 1
        grid = self.map_frustum_idxs_to_voxel_grid(intrinsics_scaled, grid_dims, fv_img_size_i, dtype=feat_voxel_i_warped.dtype, device=feat_voxel_i_warped.device)
        feat_voxel_i_warped_persp = F.grid_sample(feat_voxel_i_warped, grid, padding_mode="zeros")
        return feat_voxel_i_warped_persp
