from math import ceil
import numpy as np
import kornia
import torch
import torch.nn.functional as F

from skyeye.utils.parallel import PackedSequence
from skyeye.utils.sequence import pack_padded_images

class SemanticSegLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, ohem=None, class_weights=None, ignore_index=255, ignore_labels=None):
        if ohem is not None and (ohem <= 0 or ohem > 1):
            raise ValueError("ohem should be in (0, 1]")
        self.ohem = ohem
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.ignore_labels = ignore_labels


    def __call__(self, sem_logits, sem):
        sem_loss = []
        for i, (sem_logits_i, sem_i) in enumerate(zip(sem_logits, sem)):
            if self.ignore_labels is not None:
                sem_i[(sem_i == self.ignore_labels).any(-1)] = self.ignore_index  # Remap the ignore_labels to ignore_index

            if self.class_weights is not None:
                sem_loss_i = F.cross_entropy(sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0),
                                                      weight=torch.tensor(self.class_weights, device=sem_i.device),
                                                      ignore_index=self.ignore_index, reduction="none")
            else:
                sem_loss_i = F.cross_entropy(sem_logits_i.unsqueeze(0), sem_i.unsqueeze(0),
                                             ignore_index=self.ignore_index, reduction="none")

            sem_loss_i = sem_loss_i.view(-1)

            if self.ohem is not None and self.ohem != 1:
                top_k = int(ceil(sem_loss_i.numel() * self.ohem))
                if top_k != sem_loss_i.numel():
                    sem_loss_i, _ = sem_loss_i.topk(top_k)

            sem_loss.append(sem_loss_i.mean())

        return sum(sem_loss) / len(sem_logits)


class SemanticSegAlgo:
    def __init__(self, loss, num_classes, ignore_index=255, bev_Z_out=None, bev_W_out=None):
        self.loss = loss
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if bev_Z_out is not None:
            self.bev_Z_out = int(bev_Z_out)
        if bev_W_out is not None:
            self.bev_W_out = int(bev_W_out)

    @staticmethod
    def _pack_logits(sem_logits, valid_size, img_size):
        sem_logits = F.interpolate(sem_logits, size=img_size, mode="bilinear", align_corners=False)
        return pack_padded_images(sem_logits, valid_size)

    def _confusion_matrix(self, sem_pred, sem):
        confmat = sem[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.float)

        for sem_pred_i, sem_i in zip(sem_pred, sem):
            valid = sem_i != self.ignore_index
            if valid.any():
                sem_pred_i = sem_pred_i[valid]
                sem_i = sem_i[valid]

                confmat.index_add_(0, sem_i.view(-1) * self.num_classes + sem_pred_i.view(-1), confmat.new_ones(sem_i.numel()))

        return confmat.view(self.num_classes, self.num_classes)

    @staticmethod
    def _bev_logits_and_height(head, x):
        sem_logits, height = head(x)
        return sem_logits, height

    @staticmethod
    def _logits(head, x):
        sem_logits = head(x)
        return sem_logits

    def compute_semantic_logits(self, head, x):
        bev_sem_logits, bev_height = SemanticSegAlgo._bev_logits_and_height(head, x)
        bev_valid_size_curr = [bev_sem_logits.shape[-2:] for _ in range(bev_sem_logits.shape[0])]
        bev_img_size_curr = bev_sem_logits.shape[-2:]

        packed_sem_logits = SemanticSegAlgo._pack_logits(bev_sem_logits, bev_valid_size_curr, bev_img_size_curr)
        return packed_sem_logits, bev_height


    def training_fv(self, head, voxel_feat, sem_gt, valid_size, img_size):
        # Compute logits and prediction
        sem_logits_ = self._logits(head, voxel_feat)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)

        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])

        # Compute loss and confusion matrix
        sem_loss = self.loss(sem_logits, sem_gt)
        conf_mat = self._confusion_matrix(sem_pred, sem_gt)

        return sem_loss, conf_mat, sem_pred, sem_logits

    def inference_fv(self, head, voxel_feat, valid_size, img_size):
        sem_logits_ = self._logits(head, voxel_feat)
        sem_logits = self._pack_logits(sem_logits_, valid_size, img_size)

        sem_pred = PackedSequence([sem_logits_i.max(dim=0)[1] for sem_logits_i in sem_logits])
        return sem_pred, sem_logits

    def training_bev(self, head, voxel_feat, sem_pseudo_gt, bev_img_size=None):
        bev_sem_logits, bev_height = self.compute_semantic_logits(head, voxel_feat)

        if bev_img_size is not None:
            bev_sem_logits = F.interpolate(bev_sem_logits, size=bev_img_size, mode="bilinear", align_corners=False)

        # Compute the BEV prediction using the given logits
        bev_sem_pred = PackedSequence([bev_sem_logits_i.max(dim=0)[1] for bev_sem_logits_i in bev_sem_logits])

        # Compute loss and confusion matrix
        bev_sem_loss = self.loss(bev_sem_logits, sem_pseudo_gt)

        return bev_sem_loss, bev_sem_pred, bev_sem_logits, bev_height

    def inference_bev(self, head, x, fv_intrinsics):
        bev_sem_logits, bev_height = self.compute_semantic_logits(head, x, fv_intrinsics)
        bev_sem_pred = PackedSequence([bev_sem_logits_i.max(dim=0)[1] for bev_sem_logits_i in bev_sem_logits])
        return bev_sem_pred, bev_sem_logits, bev_height

    def compute_bev_metrics_with_gt(self, bev_sem_pred, bev_sem_gt):
        # Compute the loss wrt GT and the confusion matrix wrt GT
        conf_mat = self._confusion_matrix(bev_sem_pred, bev_sem_gt)

        return conf_mat