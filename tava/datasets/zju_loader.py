# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import cv2
import numpy as np
import torch
from tava.datasets.abstract import CachedIterDataset
from tava.datasets.zju_parser import SubjectParser
from tava.utils.camera import generate_rays, transform_cameras
from tava.utils.structures import Bones, Cameras, namedtuple_map
from tava.utils.transforms import axis_angle_to_matrix, matrix_to_rotation_6d


def _dataset_view_split(parser, split):
    _train_camera_ids = [0, 6, 12, 18]
    if split == "all":
        camera_ids = parser.camera_ids
    elif split in ["train","train_val10"]:
        camera_ids = _train_camera_ids
    elif split in ["val_ind", "val_ood", "val_view"]:
        camera_ids = list(set(parser.camera_ids) - set(_train_camera_ids))
    elif split == "test":
        camera_ids = [0]
    elif "overfit-" in split:
        i = split.find('camera:')+len('camera:')
        j = split.find('-frame:')
        if ',' in split[i:j]:
            camera_ids = split[i:j].split(',')
        elif ':' in split[i:j]:
            s, t = split[i:j].split('::')[0].split(':')
            camera_ids  = list(range(int(s),int(t)))
            if '::' in split[i:j]:
                step = int(split[i:j].split('::')[1])
                camera_ids = camera_ids[::step]
        else:
            camera_ids = [split[i:j]]
        # print(split, camera_ids)
        camera_ids = [int(ci) for ci in camera_ids]
    else:
        raise ValueError
    return camera_ids


def _dataset_frame_split(parser, split):
    if split in ["train", "val_view",'train_val10'] or "overfit-" in split:
        splits_fp = os.path.join(parser.root_dir, "splits/train.txt")
    else:
        splits_fp = os.path.join(parser.root_dir, f"splits/{split}.txt")
    with open(splits_fp, mode="r") as fp:
        frame_list = np.loadtxt(fp, dtype=int).tolist()
        if split=='train_val10':
            frame_list = frame_list[0::len(frame_list)//10]
        if "overfit-" in split:
            i = split.find('frame:')+len('frame:')         
            if ',' in split[i:]:
                frame_ids = split[i:].split(',')
            elif ':' in split[i:]:
                s, t = split[i:].split('::')[0].split(':')
                frame_ids  = list(range(int(s),int(t)))
                if '::' in split[i:]:
                    step = int(split[i:].split('::')[1])
                    frame_ids = frame_ids[::step]
            else:
                frame_ids = [int(split[i:])]
            frame_list = [frame_list[int(ii)%len(frame_list)] for ii in frame_ids]
            print(split, 'frame_list:', frame_list)
    return frame_list


def _dataset_index_list(parser, split, mode, render_view_number):
    frame_list = _dataset_frame_split(parser, split)
    if mode=='render_path':
        assert len(frame_list)==1, frame_list
        camera_ids = [0]*render_view_number
    else:
        camera_ids = _dataset_view_split(parser, split)
    index_list = []
    for frame_id in frame_list:
        index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
    return index_list


class SubjectLoader(CachedIterDataset):
    """Single subject data loader for training and evaluation."""

    SPLIT = ["all", "train", "val_ind", "val_ood", "val_view", "test",
            "train_val10"]

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        mode: str,
        resize_factor: float = 1.0,
        color_bkgd_aug: str = None,
        num_rays: int = None,
        cache_n_repeat: int = 0,
        near: float = None,
        far: float = None,
        legacy: bool = False,
        render_view_number: int = 0,
        **kwargs,
    ):  
        assert (split in self.SPLIT) or "overfit-" in split, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.resize_factor = resize_factor
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.legacy = legacy
        self.mode = mode
        self.training = (num_rays is not None) and (mode=='train')
        self.color_bkgd_aug = color_bkgd_aug
        self.parser = SubjectParser(subject_id=subject_id, root_fp=root_fp)
        self.render_view_number = render_view_number
        self.index_list = _dataset_index_list(self.parser, split, self.mode, self.render_view_number)
        self.dtype = torch.get_default_dtype()
        super().__init__(self.training, cache_n_repeat)

    def __len__(self):
        return len(self.index_list)

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        image, alpha = torch.split(rgba, [3, 1], dim=-1)
        mask = (alpha == 0) | (alpha == 1)
        if os.environ.get('SAVE_MASK','False').lower()=='true':
            from PIL import Image
            mask_img = (mask.numpy()*255).astype(np.uint8)
            mask_img = np.tile(mask_img,(1,1,3))
            mask_img = Image.fromarray(mask_img)
            save_path = os.path.join(
                    self.parser.mask_dir,
                    self.parser.image_files[data['meta_id'], data['camera_id']].replace(".jpg", "_edge.png"),
                )
            print('Save mask as', save_path)
            mask_img.save(save_path)
        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, dtype=rgba.dtype)
        else:
            # just use black during inference
            color_bkgd = torch.zeros(3, dtype=rgba.dtype)

        # only replace regions with `alpha == 0` to `color_bkgd`
        image = image * (alpha != 0) + color_bkgd * (alpha == 0)

        if self.num_rays is not None:  # usually this is in the training phase
            resolution = image.shape[0] * image.shape[1]
            # only sample rays in regions with `alpha == 0 or 1`
            indices = torch.where(
                ((alpha == 0) | (alpha == 1)).reshape(resolution)
            )[0]
            ray_indices = indices[torch.randperm(len(indices))][: self.num_rays]
            pixels = image.reshape(resolution, 3)[ray_indices]
            rays = namedtuple_map(
                lambda r: r.reshape([resolution] + list(r.shape[2:])), rays
            )
            rays = namedtuple_map(lambda x: x[ray_indices], rays)
        else:
            pixels = image

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            "mask": mask,
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        frame_id, camera_id = self.index_list[index]
        
        if self.mode == 'render_path':
            K = self.parser.cameras[0]["K"].copy()
            D = self.parser.cameras[0]["D"].copy()

            w2c_0 = self.parser.cameras[0]["w2c"].copy()
            angle = (np.math.pi*2/self.render_view_number)*index
            axis, center = self.parser.cameras["axis"], self.parser.cameras["center"]
            R = axis_angle_to_matrix(torch.tensor(axis*angle)).numpy().transpose()
            G = np.identity(4)
            G[:3,:3] = R
            G[:3,-1:] = (np.identity(3)-R).dot(center[...,None])
            w2c = w2c_0.dot(G)
        else:
            K = self.parser.cameras[camera_id]["K"].copy()
            w2c = self.parser.cameras[camera_id]["w2c"].copy()
            D = self.parser.cameras[camera_id]["D"].copy()

        # create pixels
        rgba = np.concatenate(
            [
                self.parser.load_image(frame_id, camera_id),
                self.parser.load_mask(frame_id, camera_id, trimap=True)[
                    ..., None
                ],
            ],
            axis=-1,
        )
        rgba = (
            torch.from_numpy(
                cv2.resize(
                    cv2.undistort(rgba, K, D),
                    (0, 0),
                    fx=self.resize_factor,
                    fy=self.resize_factor,
                    interpolation=cv2.INTER_AREA,
                )
            ).to(self.dtype)
            / 255.0
        )

        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(w2c).to(self.dtype),
            distorts=None,
            width=self.parser.WIDTH,
            height=self.parser.HEIGHT,
        )

        cameras = transform_cameras(cameras, self.resize_factor)
        rays = generate_rays(
            cameras, opencv_format=True, near=self.near, far=self.far
        )

        return {
            "subject_id": self.parser.subject_id,
            "camera_id": camera_id,
            # `meta_id` is used to query pose info from `pose_meta_info`
            "meta_id": frame_id,
            "rgba": rgba,  # [h, w, 4]
            "rays": rays,  # [h, w]
            "rigid_clusters": torch.tensor(
                self.parser.RIGID_BONE_IDS
            ).long(),  # bone cluster ids
        }

    def build_pose_meta_info(self):
        # create indexing for this split
        meta_ids = [frame_id for frame_id, _ in self.index_list]
        meta_data = self.parser.load_meta_data(frame_ids=meta_ids)

        # load canonical meta info.
        rest_matrixs = meta_data["rest_tfs_bone"][1:]  # [23, 4, 4]
        rest_tails = meta_data["rest_joints"][
            [
                self.parser.JOINT_NAMES.index(tail_name)
                for _, tail_name in self.parser.BONE_NAMES
            ]
        ]  # [23, 3]
        bones_rest = Bones(
            heads=None,
            tails=torch.from_numpy(rest_tails).to(self.dtype),
            transforms=torch.from_numpy(rest_matrixs).to(self.dtype),
        )  # real bones [23,]

        # load view space meta info.
        pose_matrixs = meta_data["tf_bones"][:, 1:]  # [N, 23, 4, 4]
        pose_tails = meta_data["joints"][
            :,
            [
                self.parser.JOINT_NAMES.index(tail_name)
                for _, tail_name in self.parser.BONE_NAMES
            ],
        ]  # [N, 23, 3]
        bones_posed = [
            Bones(
                heads=None,
                tails=torch.from_numpy(pose_tails[i]).to(self.dtype),
                transforms=torch.from_numpy(pose_matrixs[i]).to(self.dtype),
            )
            for i in range(len(meta_ids))
        ]  # [23,] * N
        if self.legacy:
            # The paper uses axis-angles as pose latent. However it can
            # be discontinuous.
            pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
        else:
            # An improved version is to use the 6D rotation? (not verified)
            pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
            _aa, _g_aa, _g_transl = torch.split(pose_latent, [72, 3, 3], dim=-1)
            assert (
                _aa[:, 0:3] == 0
            ).all()  # ZJU diable root rotation by default.
            _aa[
                :, 0:3
            ] = _g_aa  # write the global rotation into the root rotation.
            pose_latent = torch.cat(
                [
                    matrix_to_rotation_6d(
                        axis_angle_to_matrix(_aa.reshape(-1, 24, 3))
                    ).reshape(-1, 24 * 6),
                    _g_transl,
                ],
                dim=-1,
            )
        return {
            "meta_ids": meta_ids,
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
            "pose_latent": pose_latent,
        }
