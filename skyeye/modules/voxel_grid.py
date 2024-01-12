import torch
import torch.nn as nn
from torch.nn import functional as F
from inplace_abn import ABN


class DenseVoxelGrid(nn.Module):
    def __init__(self, in_channels=160, hidden_channels=128, dilation=6, bev_params=None, bev_W_out=None, bev_Z_out=None,
                 y_extents=None, feat_scale=None, min_level=0, levels=4, norm_act_2d=ABN, norm_act_3d=ABN):
        super(DenseVoxelGrid, self).__init__()
        self.min_level = min_level
        self.levels = levels

        # Merge the multi-scale features to generate one feature map
        self.output_1 = nn.ModuleList([
            self._DPC(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act_2d) for _ in range(levels - 2)
        ])
        self.output_2 = nn.ModuleList([
            self._3x3box(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act_2d) for _ in range(2)
        ])
        self.pre_process = nn.ModuleList([
            self._3x3box(self._seperable_conv, 128, 128, dilation, norm_act_2d) for _ in range(2)
        ])
        self.conv_cat_out = nn.Conv2d(512, 128, 1, padding=0)

        self.bev_Z_out = int(bev_Z_out * feat_scale)
        self.depth_network = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                           norm_act_2d(256),
                                           nn.Conv2d(256, 256, 3, padding=1),
                                           norm_act_2d(256),
                                           nn.Conv2d(256, self.bev_Z_out, 1, padding=0))    # Note: 1 Channel per pixel in the BEV space

        # Things to generate the voxel grid
        self.feat_scale = feat_scale
        self.bev_Z_out = int(bev_Z_out * feat_scale)
        self.bev_W_out = int(bev_W_out * feat_scale)
        self.resolution = bev_params['cam_z'] / bev_params['f'] / feat_scale
        self.voxel_extents = [-(self.bev_W_out * self.resolution / 2), y_extents[0], 0,
                              (self.bev_W_out * self.resolution / 2), y_extents[1], self.bev_Z_out * self.resolution]
        self.voxel_index_grid = self._make_voxel_index_grid()

        self.conv3d_1 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1),
                                      norm_act_3d(64))
        self.conv3d_2 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1),
                                      norm_act_3d(64))


    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(DenseVoxelGrid._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation, padding=dilation,
                                       groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    class _3x3box(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act):
            super(DenseVoxelGrid._3x3box, self).__init__()

            self.conv1_3x3_1 = seperable_conv(in_channels, out_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(out_channels)
            self.conv1_3x3_2 = seperable_conv(out_channels, out_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(out_channels)

        def forward(self, x):
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            return x

    class _DPC(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act):
            super(DenseVoxelGrid._DPC, self).__init__()

            self.conv1_3x3_1 = seperable_conv(in_channels, in_channels, (1, 6), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(in_channels)
            self.conv1_3x3_2 = seperable_conv(in_channels, in_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(in_channels)
            self.conv1_3x3_3 = seperable_conv(in_channels, in_channels, (6, 21), norm_act, bias=False)
            self.conv1_3x3_3_bn = norm_act(in_channels)
            self.conv1_3x3_4 = seperable_conv(in_channels, in_channels, (18, 15), norm_act, bias=False)
            self.conv1_3x3_4_bn = norm_act(in_channels)
            self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6, 3), norm_act, bias=False)
            self.conv1_3x3_5_bn = norm_act(in_channels)

            self.conv2 = nn.Conv2d(in_channels * 5, out_channels, 1, bias=False)
            self.bn2 = norm_act(out_channels)

        def forward(self, x):
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x1 = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            x2 = self.conv1_3x3_3_bn(self.conv1_3x3_3(x))
            x3 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x))
            x4 = self.conv1_3x3_5_bn(self.conv1_3x3_5(x3))
            x = torch.cat([x, x1, x2, x3, x4], dim=1)
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def merge_multiscale_features(self, ms_feat):
        """Merge the multiscale features to obtain one big feature map"""
        ms_feat = list(ms_feat[self.min_level:self.min_level + self.levels])

        ref_size = ms_feat[0].shape[-2:]

        i = self.min_level + self.levels - 1
        js = 0
        for output in self.output_1:
            ms_feat[i] = output(ms_feat[i])
            i = i - 1
        interm = self.pre_process[js](ms_feat[i + 1] + F.interpolate(ms_feat[i + 2], size=ms_feat[i + 1].shape[-2:],
                                                                     mode="bilinear", align_corners=False))
        for output in self.output_2:
            ms_feat[i] = output(ms_feat[i])
            if js == 1:
                interm = self.pre_process[js](ms_feat[i + 1])

            ms_feat[i] = ms_feat[i] + F.interpolate(interm, size=ms_feat[i].shape[-2:], mode="bilinear",
                                                    align_corners=False)
            js += 1
            i = i - 1
        for i in range(self.min_level, self.min_level + self.levels):
            if i > 0:
                ms_feat[i] = F.interpolate(ms_feat[i], size=ref_size, mode="bilinear", align_corners=False)

        feat_merged = torch.cat(ms_feat, dim=1)
        feat_merged_small = self.conv_cat_out(feat_merged)
        return feat_merged, feat_merged_small

    def _get_depth_dist(self, depth_feat):
        return depth_feat.softmax(dim=1)

    def get_voxel_features(self, feat_merged_2d, sample_pts_2d, voxel_idxs, voxel_dims):
        feat_voxel = torch.zeros([feat_merged_2d.shape[0], feat_merged_2d.shape[1], voxel_dims[0], voxel_dims[1], voxel_dims[2]], device=feat_merged_2d.device)
        for i, (sample_pts_2d_i, voxel_idxs_i) in enumerate(zip(sample_pts_2d, voxel_idxs)):
            # Normalise the sampling coordinates. This is still in W, H format
            H_min, H_max, W_min, W_max = 0, feat_merged_2d.shape[2], 0, feat_merged_2d.shape[3]
            sample_pts_2d_i[0] = (sample_pts_2d_i[0] - W_min) / (W_max - W_min) * 2 - 1
            sample_pts_2d_i[1] = (sample_pts_2d_i[1] - H_min) / (H_max - H_min) * 2 - 1

            # Bring it to H, W format
            sample_pts_2d_permute_i = torch.zeros_like(sample_pts_2d_i)
            sample_pts_2d_permute_i[0] = sample_pts_2d_i[1]
            sample_pts_2d_permute_i[1] = sample_pts_2d_i[0]
            sample_pts_2d_permute_i = torch.transpose(sample_pts_2d_permute_i, 0, 1).unsqueeze(0).unsqueeze(0)

            feat_3d_i = F.grid_sample(feat_merged_2d[i].unsqueeze(0), sample_pts_2d_permute_i)

            feat_voxel[i, :, voxel_idxs_i[2], voxel_idxs_i[1], voxel_idxs_i[0]] = feat_3d_i.squeeze(2).squeeze(0)

        return feat_voxel

    def _make_voxel_index_grid(self):
        x1, y1, z1, x2, y2, z2 = self.voxel_extents
        z, y, x = torch.arange(z1, z2, self.resolution), \
                     torch.arange(y1, y2, self.resolution), \
                     torch.arange(x1, x2, self.resolution)
        x_grid, y_grid, z_grid = torch.meshgrid(x, y, z)
        grid = torch.stack((x_grid, y_grid, z_grid), dim=0)

        return grid

    def voxel_grid_to_fv(self, intrinsics, fv_img_shape, batch_size, device):
        """ Generate the mapping between the voxel grid indices and the FV feature indices. This will be used to create the voxel grid features from the FV. """

        # Scale the intrinsics matrix to match the current feature scale
        intrinsics_scale = intrinsics * self.feat_scale
        intrinsics_scale[:, 2, 2] = 1

        # Get the shape of the feature map
        feat_shape = (fv_img_shape[0] * self.feat_scale, fv_img_shape[1] * self.feat_scale)

        # Map the values from the voxel grid to the FV feature space using perspective projection
        voxel_coords = torch.stack([self.voxel_index_grid] * batch_size, dim=0).to(device)

        x3d = voxel_coords[:, 0, :, :, :].view(batch_size, -1)
        y3d = voxel_coords[:, 1, :, :, :].view(batch_size, -1)
        z3d = voxel_coords[:, 2, :, :, :].clamp(min=1e-5).view(batch_size, -1)

        # Compute the pixel coordinates
        u2d = (x3d * intrinsics_scale[:, 0, 0].unsqueeze(1) / z3d) + intrinsics_scale[:, 0, 2].unsqueeze(1)
        v2d = (y3d * intrinsics_scale[:, 1, 1].unsqueeze(1) / z3d) + intrinsics_scale[:, 1, 2].unsqueeze(1)

        # Put the u2d and v2d vectors together and reshape them
        u2d_norm = ((u2d / (feat_shape[1] - 1)) - 0.5) * 2  # Normalise to [-1, 1]
        v2d_norm = ((v2d / (feat_shape[0] - 1)) - 0.5) * 2  # Normalise to [-1, 1]

        feat_coords_map = torch.stack([u2d_norm, v2d_norm], dim=2)  # dim: batch_size, H_bevxW_bev, 2
        feat_coords_map = feat_coords_map.view(batch_size, voxel_coords.shape[2], voxel_coords.shape[3], voxel_coords.shape[4], 2)
        feat_coords_map = feat_coords_map.permute(0, 3, 2, 1, 4)

        return feat_coords_map

    def unproj_depth_dist(self, intrinsics, fv_img_shape, batch_size, device):
        # Scale the intrinsics matrix to match the current feature scale
        intrinsics_scale = intrinsics * self.feat_scale
        intrinsics_scale[:, 2, 2] = 1

        # Get the shape of the feature map
        feat_shape = (fv_img_shape[0] * self.feat_scale, fv_img_shape[1] * self.feat_scale)

        # Map the values from the voxel grid to the FV feature space using perspective projection
        voxel_coords = torch.stack([self.voxel_index_grid] * batch_size, dim=0).to(device)

        x3d = voxel_coords[:, 0, :, :, :].view(batch_size, -1)
        y3d = voxel_coords[:, 1, :, :, :].view(batch_size, -1)
        z3d = voxel_coords[:, 2, :, :, :].clamp(min=1e-5).view(batch_size, -1)

        # Compute the pixel coordinates
        u2d = (x3d * intrinsics_scale[:, 0, 0].unsqueeze(1) / z3d) + intrinsics_scale[:, 0, 2].unsqueeze(1)
        v2d = (y3d * intrinsics_scale[:, 1, 1].unsqueeze(1) / z3d) + intrinsics_scale[:, 1, 2].unsqueeze(1)
        z = z3d

        # Put the u2d and v2d vectors together and reshape them
        u2d_norm = ((u2d / (feat_shape[1] - 1)) - 0.5) * 2  # Normalise to [-1, 1]
        v2d_norm = ((v2d / (feat_shape[0] - 1)) - 0.5) * 2  # Normalise to [-1, 1]
        z_norm = ((z / (voxel_coords.shape[4] - 1)) - 0.5) * 2

        depth_coords_map = torch.stack([u2d_norm, v2d_norm, z_norm], dim=2)  # dim: batch_size, H_bevxW_bev, 2
        depth_coords_map = depth_coords_map.view(batch_size, voxel_coords.shape[2], voxel_coords.shape[3], voxel_coords.shape[4], 3)
        depth_coords_map = depth_coords_map.permute(0, 3, 2, 1, 4)

        return depth_coords_map

    def forward(self, ms_feat, intrinsics, fv_img_shape):
        # Merge the multiscale feature maps to get one merged representation
        # For now, we use the merging technique as used in EfficientPS
        feat_merged_2d, feat_merged_2d_small = self.merge_multiscale_features(ms_feat)

        feat_coords_map = self.voxel_grid_to_fv(intrinsics, fv_img_shape, batch_size=feat_merged_2d_small.shape[0], device=feat_merged_2d_small.device)
        B, D, H, W, _ = feat_coords_map.shape
        ones = (torch.ones((B, D, H, W, 1)) * -1).to(feat_coords_map.device)
        feat_coords_map_3d = torch.cat([feat_coords_map, ones], dim=-1)
        feat_merged_2d_small_3d = feat_merged_2d_small.unsqueeze(2)
        feat_voxel_3d = F.grid_sample(feat_merged_2d_small_3d, feat_coords_map_3d, padding_mode="zeros")

        # Predict implicit depth and unproject it to the voxel grid
        depth_feat = self.depth_network(feat_merged_2d)
        depth_dist = self._get_depth_dist(depth_feat)

        coords_depth_dist = self.unproj_depth_dist(intrinsics, fv_img_shape, batch_size=feat_merged_2d_small.shape[0], device=feat_merged_2d_small.device)
        depth_dist_unproj = F.grid_sample(depth_dist.unsqueeze(1), coords_depth_dist, padding_mode="zeros")

        feat_voxel_3d = feat_voxel_3d * depth_dist_unproj

        # Process the 3D features and return the voxel grid
        feat_voxel_3d = self.conv3d_1(feat_voxel_3d)
        feat_voxel_3d = self.conv3d_2(feat_voxel_3d)

        return feat_voxel_3d, feat_merged_2d, depth_dist, depth_dist_unproj, feat_coords_map_3d