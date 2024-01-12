import torch
import torch.nn as nn
import torch.nn.functional as F

class FVSemHead(nn.Module):
    def __init__(self, num_classes):
        super(FVSemHead, self).__init__()
        self.num_classes = num_classes
        self.conv2d_sem = nn.Conv2d(64, num_classes, 1)


    def forward(self, feat_voxel):
        # Flatten the voxel grid along the depth axis
        feat_2d = torch.mean(feat_voxel, dim=2)  # Shape: B, C, H, W
        fv_sem_logits = self.conv2d_sem(feat_2d)

        return fv_sem_logits


class BEVSemHead(nn.Module):
    def __init__(self, num_classes, bev_W_out=None, bev_Z_out=None):
        super(BEVSemHead, self).__init__()
        self.num_classes = num_classes

        self.bev_Z_out = int(bev_Z_out)
        self.bev_W_out = int(bev_W_out)

        self.conv2d_sem_final = nn.Conv2d(64, num_classes, 1)


    def forward(self, feat_voxel):
        # Flatten the voxel grid along the height dimension
        feat_voxel = torch.mean(feat_voxel, dim=3) # Shape: B, C, D, W

        # Now decode the semantic features with multiple layers
        bev_sem_logits = self.conv2d_sem_final(feat_voxel)
        bev_sem_logits = F.interpolate(bev_sem_logits, (self.bev_Z_out, self.bev_W_out), mode="bilinear", align_corners=False)
        bev_sem_logits = torch.flip(bev_sem_logits, dims=[3])
        bev_sem_logits = torch.rot90(bev_sem_logits, k=1, dims=[2, 3])

        # The height map rotation is handled directly when the pseudo labels are created. No need to do anything here.
        bev_base_height = torch.ones(bev_sem_logits.shape[0], 1, self.bev_Z_out, self.bev_W_out).to(bev_sem_logits.device) * 1.55
        bev_base_height = torch.flip(bev_base_height, dims=[3])

        return bev_sem_logits, bev_base_height