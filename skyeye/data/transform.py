import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as tfn
from torchvision.transforms import transforms
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F


################### START - COPY FROM OLDER PYTORCH VERSION FOR BACKWARD COMPATIBILITY ###################
class Lambda:
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    """Composes several transforms together. This transform does not support torchscript. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

################### END - COPY FROM OLDER PYTORCH VERSION FOR BACKWARD COMPATIBILITY ###################


class BEVTransform:
    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None,
                 front_resize=None,
                 bev_centre_crop=None,
                 bev_crop=None,
                 scale=None,
                 random_flip=False,
                 random_brightness=None,
                 random_contrast=None,
                 random_saturation=None,
                 random_hue=None):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.front_resize = front_resize
        self.bev_centre_crop = bev_centre_crop
        self.bev_crop = bev_crop
        self.scale = scale
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.random_flip = random_flip
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.random_hue = random_hue

    def _scale(self, img, bev_msk, bev_plabel, front_msk, weights_msk):
        # Scale the image and the mask
        if img is not None:
            in_img_w, in_img_h = img[0].size[0], img[0].size[1]
            out_img_w, out_img_h = int(in_img_w * self.scale), int(in_img_h * self.scale)
            img = [rgb.resize((out_img_w, out_img_h)) for rgb in img]

        if bev_msk is not None:
            in_msk_w, in_msk_h = bev_msk[0].size[0], bev_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            bev_msk = [m.resize((out_msk_w, out_msk_h), Image.NEAREST) for m in bev_msk]

        if bev_plabel is not None:
            in_msk_w, in_msk_h = bev_plabel[0].size[0], bev_plabel[0].size[1]
            out_plabel_w, out_plabel_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            bev_plabel = [m.resize((out_plabel_w, out_plabel_h), Image.NEAREST) for m in bev_plabel]

        if front_msk is not None:
            in_msk_w, in_msk_h = front_msk[0].size[0], front_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            front_msk = [m.resize((out_msk_w, out_msk_h), Image.NEAREST) for m in front_msk]

        if weights_msk is not None:
            in_msk_w, in_msk_h = weights_msk[0].size[0], weights_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            weights_msk = [m.resize((out_msk_w, out_msk_h), Image.BILINEAR) for m in weights_msk]

        return img, bev_msk, bev_plabel, front_msk, weights_msk

    def _resize(self, img, mode):
        if img is not None:
            # Resize the image
            out_img_w, out_img_h = self.front_resize[1], self.front_resize[0]
            img = [rgb.resize((out_img_w, out_img_h), mode) for rgb in img]

        return img

    def _centre_crop(self, msk):
        if msk is not None:
            ip_height, ip_width = msk[0].size[1], msk[0].size[0]

            # Check that the crop dimensions are not larger than the input dimensions
            if self.bev_centre_crop[0] > ip_height or self.bev_centre_crop[1] > ip_width:
                raise RuntimeError("Crop dimensions need to be smaller than the input dimensions."
                                   "Crop: {}, Input: {}".format(self.bev_centre_crop, (ip_height, ip_width)))

            # We want to crop from the centre
            min_row = ip_height // 2 - self.bev_centre_crop[0] // 2
            max_row = ip_height // 2 + self.bev_centre_crop[0] // 2
            min_col = ip_width // 2 - self.bev_centre_crop[1] // 2
            max_col = ip_width // 2 + self.bev_centre_crop[1] // 2

            # (Left, Top, Right, Bottom)
            msk_cropped = [m.crop((min_col, min_row, max_col, max_row)) for m in msk]
            return msk_cropped
        else:
            return msk

    def _crop(self, msk):
        if msk is not None:
            ip_height, ip_width = msk[0].size[1], msk[0].size[0]

            # Check that the crop dimensions are not larger than the input dimensions
            if self.bev_crop[0] > ip_height or self.bev_crop[1] > ip_width:
                raise RuntimeError("Crop dimensions need to be smaller than the input dimensions."
                                   "Crop: {}, Input: {}".format(self.bev_crop, (ip_height, ip_width)))

            # We want to crop from the centre
            min_row = 0
            max_row = self.bev_crop[0]
            min_col = 0 #self.bev_crop[1]
            max_col = ip_width

            # (Left, Top, Right, Bottom)
            msk_cropped = [m.crop((min_col, min_row, max_col, max_row)) for m in msk]
            return msk_cropped
        else:
            return msk

    @staticmethod
    def _random_flip(img, bev_msk, bev_plabel, front_msk, weights_msk, transform_dict):
        if random.random() < 0.5:
            transform_dict['flip'] = True

            # Horizontally flip the RGB image and the front mask
            if img is not None:
                img = [rgb.transpose(Image.FLIP_LEFT_RIGHT) for rgb in img]
            if front_msk is not None:
                front_msk = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in front_msk]

            # Flip the BEV panoptic mask. The mask is sideways, so that mask has to be flipped top-down
            if bev_msk is not None:
                bev_msk = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in bev_msk]
            if bev_plabel is not None:
                bev_plabel = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in bev_plabel]
            if weights_msk is not None:
                weights_msk = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in weights_msk]

            return img, bev_msk, bev_plabel, front_msk, weights_msk, transform_dict
        else:
            transform_dict['flip'] = False
            return img, bev_msk, bev_plabel, front_msk, weights_msk, transform_dict

    def _normalize_image(self, img):
        if img is not None:
            if (self.rgb_mean is not None) and (self.rgb_std is not None):
                img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
                img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    @staticmethod
    def _compact_labels(msk, cat, iscrowd):
        ids = np.unique(msk)
        if 0 not in ids:
            ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

        ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
        ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)

        msk = ids_to_compact[msk]
        cat = cat[ids]
        iscrowd = iscrowd[ids]

        return msk, cat, iscrowd

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, bev_msk=None, bev_plabel=None, fv_msk=None, bev_weights_msk=None,
                 bev_cat=None, bev_iscrowd=None, fv_cat=None, fv_iscrowd=None,
                 fv_intrinsics=None, ego_pose=None):
        # Stores whether a transformation was performed or not
        transform_status = {}

        # Random flip
        if self.random_flip:
            img, bev_msk, bev_plabel, fv_msk, bev_weights_msk, transform_status = self._random_flip(img, bev_msk, bev_plabel, fv_msk,bev_weights_msk, transform_status)
        else:
            transform_status['flip'] = False

        # Crop the mask from the centre to the size of the crop
        # Weights mask is already cropped for us :D
        if self.bev_centre_crop:
            bev_msk = self._centre_crop(bev_msk)
            bev_plabel = self._centre_crop(bev_plabel)
            transform_status['bev_centre_crop'] = self.bev_centre_crop
        else:
            transform_status['bev_centre_crop'] = False

        if self.bev_crop:
            bev_msk = self._crop(bev_msk)
            bev_plabel = self._crop(bev_plabel)
            transform_status['bev_crop'] = self.bev_crop
        else:
            transform_status['bev_crop'] = False

        # Resize the RGB image and the front mask to the given dimension
        if self.front_resize:
            img = self._resize(img, Image.BILINEAR)
            fv_msk = self._resize(fv_msk, Image.NEAREST)
            transform_status['front_resize'] = self.front_resize
        else:
            transform_status['front_resize'] = False

        # Scale the images and the mask to a smaller value. It is a constant scale!
        if self.scale:
            img, bev_msk, bev_plabel, fv_msk, bev_weights_msk = self._scale(img, bev_msk, bev_plabel, fv_msk, bev_weights_msk)
            transform_status['scale'] = self.scale
        else:
            transform_status['scale'] = False

        # Random Colour Jitter. Apply the same colour jitter to all the images
        if (self.random_brightness is not None) and (self.random_contrast is not None) \
                and (self.random_hue is not None) and (self.random_flip is not None):
            colour_jitter = ColorJitter(brightness=self.random_brightness, contrast=self.random_contrast,
                                        saturation=self.random_saturation, hue=self.random_hue)
            colour_jitter_transform = self.get_params(colour_jitter.brightness, colour_jitter.contrast,
                                                      colour_jitter.saturation, colour_jitter.hue)
            img = [colour_jitter_transform(rgb) for rgb in img]
            colour_jitter_params = {"brightness": colour_jitter.brightness,
                                    "contrast": colour_jitter.contrast,
                                    "saturation": colour_jitter.saturation,
                                    "hue": colour_jitter.hue}
            transform_status['colour_jitter'] = colour_jitter_params
        else:
            transform_status['colour_jitter'] = False
  
        # Wrap in np.array
        if bev_cat is not None:
            bev_cat = [np.array(cat, dtype=np.int32) for cat in bev_cat]
        if bev_iscrowd is not None:
            bev_iscrowd = [np.array(iscrowd, dtype=np.uint8) for iscrowd in bev_iscrowd]
        if fv_cat is not None:
            fv_cat = [np.array(cat, dtype=np.int32) if cat is not None else None for cat in fv_cat]
        if fv_iscrowd is not None:
            fv_iscrowd = [np.array(iscrowd, dtype=np.uint8) if iscrowd is not None else None for iscrowd in fv_iscrowd]

        # Adjust calib and wrap in np.array
        if fv_intrinsics is not None:
            fv_intrinsics = [np.array(intrinsics, dtype=np.float32) for intrinsics in fv_intrinsics]
            for i in range(len(fv_intrinsics)):
                if len(fv_intrinsics[i].shape) == 3:
                    fv_intrinsics[i][:, 0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                    fv_intrinsics[i][:, 1, 1] *= float(self.front_resize[0]) / self.shortest_size
                    fv_intrinsics[i][:, 0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                    fv_intrinsics[i][:, 1, 2] *= float(self.front_resize[0]) / self.shortest_size
                    # Change the cx and cy if the image is flipped
                    if transform_status['flip']:
                        fv_intrinsics[i][:, 0, 2] = self.front_resize[1] - fv_intrinsics[i][:, 0, 2]
                else:
                    fv_intrinsics[i][0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                    fv_intrinsics[i][1, 1] *= float(self.front_resize[0]) / self.shortest_size
                    fv_intrinsics[i][0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                    fv_intrinsics[i][1, 2] *= float(self.front_resize[0]) / self.shortest_size
                    if transform_status['flip']:
                        fv_intrinsics[i][0, 2] = self.front_resize[1] - fv_intrinsics[i][0, 2]

        if ego_pose is not None:
            ego_pose = [np.array(pose, dtype=np.float32) for pose in ego_pose]

        # Image transformations
        img = [tfn.to_tensor(rgb) for rgb in img]
        img = [self._normalize_image(rgb) for rgb in img]

        # Label transformations,
        if bev_msk is not None:
            bev_msk = [np.expand_dims(np.array(m, dtype=np.int32, copy=False), axis=0) for m in bev_msk]
            for i in range(len(bev_msk)):
                bev_msk[i], bev_cat[i], bev_iscrowd[i] = self._compact_labels(bev_msk[i], bev_cat[i], bev_iscrowd[i])

        if bev_plabel is not None:
            bev_plabel = [np.array(m, dtype=np.int32, copy=False) for m in bev_plabel]
            bev_plabel = np.stack(bev_plabel, axis=0)

        if bev_weights_msk is not None:
            bev_weights_msk = [np.array(m, dtype=np.int32, copy=False) for m in bev_weights_msk]
            bev_weights_msk = np.stack(bev_weights_msk, axis=0)

        if fv_msk is not None:
            fv_msk = [np.expand_dims(np.array(m, dtype=np.int32, copy=False), axis=0) if m is not None else None for m in fv_msk]
            if fv_cat is not None and fv_iscrowd is not None:
                for i in range(len(fv_msk)):
                    fv_msk[i], fv_cat[i], fv_iscrowd[i] = self._compact_labels(fv_msk[i], fv_cat[i], fv_iscrowd[i])

        # Convert labels to torch and extract bounding boxes
        if bev_msk is not None:
            bev_msk = [torch.from_numpy(msk.astype(np.int_)) for msk in bev_msk]
        if bev_plabel is not None:
            bev_plabel = [torch.from_numpy(plabel.astype(np.int_)) for plabel in bev_plabel]
        if fv_msk is not None:
            fv_msk = [torch.from_numpy(msk.astype(np.int_)) if msk is not None else None for msk in fv_msk]
        if bev_weights_msk is not None:
            bev_weights_msk = torch.from_numpy(bev_weights_msk.astype(np.float))
        if bev_cat is not None:
            bev_cat = [torch.from_numpy(cat.astype(np.int_)) for cat in bev_cat]
        if bev_iscrowd is not None:
            bev_iscrowd = [torch.from_numpy(iscrowd) for iscrowd in bev_iscrowd]
        if fv_cat is not None:
            fv_cat = [torch.from_numpy(cat.astype(np.int_)) if cat is not None else None for cat in fv_cat]
        if fv_iscrowd is not None:
            fv_iscrowd = [torch.from_numpy(iscrowd) if iscrowd is not None else None for iscrowd in fv_iscrowd]
        if fv_intrinsics is not None:
            fv_intrinsics = [torch.from_numpy(intrinsics) for intrinsics in fv_intrinsics]
        if ego_pose is not None:
            ego_pose = [torch.from_numpy(pose) for pose in ego_pose]

        return dict(img=img, bev_msk=bev_msk, bev_plabel=bev_plabel, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                    bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd,
                    fv_intrinsics=fv_intrinsics, ego_pose=ego_pose, transform_status=transform_status)
