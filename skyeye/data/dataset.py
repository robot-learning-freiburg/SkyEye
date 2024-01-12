import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import umsgpack
from PIL import Image
import json


class BEVKitti360Dataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _BEV_PLABEL_DIR = "bev_plabel_dynamic"
    _FV_MSK_DIR = "front_msk_seam"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _PERCENTAGES_MULTIRUN_DIR = "percentages_multirun"
    _PERCENTAGES_DIR = "percentages"
    _BEV_METADATA_FILE = "metadata_ortho.bin"
    _FV_METADATA_FILE = "metadata_front.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform, window=0, bev_percentage=100, run_iter=None):
        super(BEVKitti360Dataset, self).__init__()
        self.seam_root_dir = seam_root_dir  # Directory of seamless data
        self.kitti_root_dir = dataset_root_dir  #  Directory of the KITTI360 data
        self.split_name = split_name
        self.transform = transform
        self.window = window  # Single-sided window count. The number of images samples is [i - window to i + window]
        self.rgb_cameras = ['front']
        if bev_percentage < 1:
            self.bev_percentage = bev_percentage
        else:
            self.bev_percentage = int(bev_percentage)
        self.run_iter = run_iter

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_MSK_DIR, BEVKitti360Dataset._BEV_DIR)
        self._bev_plabel_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_PLABEL_DIR, BEVKitti360Dataset._BEV_DIR)
        self._fv_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._FV_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)
        if self.run_iter is not None:
            self._percentages_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR, BEVKitti360Dataset._PERCENTAGES_MULTIRUN_DIR)
        else:
            self._percentages_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR, BEVKitti360Dataset._PERCENTAGES_DIR)

        # Load meta-data and split
        self._bev_meta, self._bev_images, self._bev_images_all, self._fv_meta, self._fv_images, self._fv_images_all,\
        self._img_map, self.bev_percent_split = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._BEV_METADATA_FILE), "rb") as fid:
            bev_metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._FV_METADATA_FILE), 'rb') as fid:
            fv_metadata = umsgpack.unpack(fid, encoding="utf-8")

        # Read the files for this split
        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        if self.split_name == "train":
            # Get all the frames in the train dataset. This will be used for generating samples for temporal consistency.
            with open(os.path.join(self._lst_dir, "{}_all.txt".format(self.split_name)), 'r') as fid:
                lst_all = fid.readlines()
                lst_all = [line.strip() for line in lst_all]

            # Get all the samples for which the BEV plabels have to be loaded.
            if self.run_iter is not None:
                percentage_file = os.path.join(self._percentages_dir, "{}_{}_{}.txt".format(self.split_name, self.bev_percentage, self.run_iter))
                print("Loading {}_{}% file".format(self.bev_percentage, self.run_iter))
            else:
                percentage_file = os.path.join(self._percentages_dir, "{}_{}.txt".format(self.split_name, self.bev_percentage))
                print("Loading {}% file".format(self.bev_percentage))
            with open(percentage_file, 'r') as fid:
                lst_percent = fid.readlines()
                lst_percent = [line.strip() for line in lst_percent]
        else:
            lst_all = lst
            lst_percent = lst

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        fv_msk_frames = os.listdir(self._fv_msk_dir)
        fv_msk_frames = [frame.split(".")[0] for frame in fv_msk_frames]
        fv_msk_frames_exist_map = {entry: True for entry in fv_msk_frames}  # This is to speed-up the dataloader
        lst = [entry for entry in lst if entry in fv_msk_frames_exist_map]
        lst_all = [entry for entry in lst_all if entry in fv_msk_frames_exist_map]
        lst_all_frame_idx_map = {entry: idx for idx, entry in enumerate(lst_all)}

        # Remove the corner scene elements so that they can satisfy the window constraint
        if self.window > 0:
            lst_filt = [entry for entry in lst
                        if (((len(lst_all) - lst_all_frame_idx_map[entry]) > self.window) and (lst_all_frame_idx_map[entry] >= self.window))
                        and ((lst_all[lst_all_frame_idx_map[entry] - self.window].split(";")[0] == entry.split(";")[0]) and (lst_all[lst_all_frame_idx_map[entry] + self.window].split(";")[0] == entry.split(";")[0]))]
            lst = lst_filt

        # Filter based on the samples plabels
        if self.bev_percentage < 100:
            lst_filt = [entry for entry in lst if entry in lst_percent]
            lst = lst_filt

        # Remove any potential duplicates
        lst = set(lst)
        lst_percent = set(lst_percent)

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        bev_meta = bev_metadata["meta"]
        bev_images = [img_desc for img_desc in bev_metadata["images"] if img_desc["id"] in lst]
        fv_meta = fv_metadata["meta"]
        fv_images = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst]

        # Check for inconsistency due to inconsistencies in the input files or dataset
        bev_images_ids = [bev_img["id"] for bev_img in bev_images]
        fv_images_ids = [fv_img["id"] for fv_img in fv_images]
        assert set(bev_images_ids) == set(fv_images_ids) and len(bev_images_ids) == len(fv_images_ids), 'Inconsistency between fv_images and bev_images detected'

        if lst_all is not None:
            bev_images_all = [img_desc for img_desc in bev_metadata['images'] if img_desc['id'] in lst_all]
            fv_images_all = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst_all]
        else:
            bev_images_all, fv_images_all = None, None

        return bev_meta, bev_images, bev_images_all, fv_meta, fv_images, fv_images_all, img_map, lst_percent

    def _find_index(self, list, key, value):
        for i, dic in enumerate(list):
            if dic[key] == value:
                return i
        return None

    def _load_item(self, item_idx):
        # Find the index of the element in the list containing all elements
        all_idx = self._find_index(self._fv_images_all, "id", self._fv_images[item_idx]['id'])
        if all_idx is None:
            raise IOError("Required index not found!")

        if self.window > 0:
            left = all_idx - self.window
            right = all_idx + self.window
            bev_img_desc_list = self._bev_images_all[left:right+1]
            fv_img_desc_list = self._fv_images_all[left:right+1]
        else:
            bev_img_desc_list = [self._bev_images[item_idx]]
            fv_img_desc_list = [self._fv_images[item_idx]]

        scene, frame_id = self._bev_images[item_idx]["id"].split(";")

        # Get the RGB file names
        img_file = [os.path.join(self.kitti_root_dir, self._img_map["front"]["{}.png".format(bev_img_desc['id'])])
                    for bev_img_desc in bev_img_desc_list]

        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Scene: {}, Frame: {}".format(scene, frame_id))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        bev_msk_file = [os.path.join(self._bev_msk_dir, "{}.png".format(bev_img_desc['id']))
                        for bev_img_desc in bev_img_desc_list]
        bev_msk = [Image.open(msk) for msk in bev_msk_file]

        # Load the plabel. In contrast to BEV gt msks, this only takes the middle element
        bev_plabel_name = bev_img_desc_list[len(bev_img_desc_list) // 2]['id']
        bev_plabel_path = os.path.join(self._bev_plabel_dir, "{}.png".format(bev_plabel_name))
        bev_plabel_file = [bev_plabel_path]
        to_pil_image = T.ToPILImage()
        if os.path.exists(bev_plabel_path) and (bev_plabel_name in self.bev_percent_split):
            bev_plabel = [Image.open(plabel).rotate(90, expand=True) for plabel in bev_plabel_file]
        else:
            bev_plabel = [to_pil_image(torch.rot90(torch.ones(size=bev_msk[0].size, dtype=torch.int32) * 255, k=1,dims=[0, 1])) for _ in bev_plabel_file]

        # Load the front mask
        fv_msk_file = [os.path.join(self._fv_msk_dir, "{}.png".format(fv_img_desc['id']))
                       for fv_img_desc in fv_img_desc_list]
        fv_msk = [Image.open(msk) for msk in fv_msk_file]

        assert len(fv_msk) == len(img), "FV Mask: {}, Img: {}".format(len(fv_img_desc_list), len(img))

        bev_weights_msk_combined = None

        # Get the other information
        bev_cat = [bev_img_desc["cat"] for bev_img_desc in bev_img_desc_list]
        bev_iscrowd = [bev_img_desc["iscrowd"] for bev_img_desc in bev_img_desc_list]
        fv_cat = [fv_img_desc['cat'] for fv_img_desc in fv_img_desc_list]
        fv_iscrowd = [fv_img_desc['iscrowd'] for fv_img_desc in fv_img_desc_list]
        fv_intrinsics = [fv_img_desc["cam_intrinsic"] for fv_img_desc in fv_img_desc_list]
        ego_pose = [fv_img_desc['ego_pose'] for fv_img_desc in fv_img_desc_list]  # This loads the cam0 pose

        # Get the ids of all the frames
        frame_ids = [bev_img_desc["id"] for bev_img_desc in bev_img_desc_list]

        return img, bev_msk, bev_plabel, fv_msk, bev_weights_msk_combined, bev_cat, bev_iscrowd, \
               fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, frame_ids

    @property
    def fv_categories(self):
        """Category names"""
        return self._fv_meta["categories"]

    @property
    def fv_num_categories(self):
        """Number of categories"""
        return len(self.fv_categories)

    @property
    def fv_num_stuff(self):
        """Number of "stuff" categories"""
        return self._fv_meta["num_stuff"]

    @property
    def fv_num_thing(self):
        """Number of "thing" categories"""
        return self.fv_num_categories - self.fv_num_stuff

    @property
    def bev_categories(self):
        """Category names"""
        return self._bev_meta["categories"]

    @property
    def bev_num_categories(self):
        """Number of categories"""
        return len(self.bev_categories)

    @property
    def bev_num_stuff(self):
        """Number of "stuff" categories"""
        return self._bev_meta["num_stuff"]

    @property
    def bev_num_thing(self):
        """Number of "thing" categories"""
        return self.bev_num_categories - self.bev_num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._fv_meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._fv_meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._fv_images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._fv_images]

    @property
    def dataset_name(self):
        return "Kitti360"

    def __len__(self):
        return len(self._fv_images)

    def __getitem__(self, item):
        img, bev_msk, bev_plabel, fv_msk, bev_weights_msk, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, bev_plabel=bev_plabel, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                             bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd, fv_intrinsics=fv_intrinsics,
                             ego_pose=ego_pose)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        for m in fv_msk:
            m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)