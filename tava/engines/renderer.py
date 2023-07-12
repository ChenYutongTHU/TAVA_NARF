# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import math
import os

import torch
import numpy as np
from hydra.utils import instantiate
from tava.engines.abstract import AbstractEngine
from tava.utils.evaluation import eval_epoch, render_image
from tava.utils.structures import namedtuple_map
import imageio

LOGGER = logging.getLogger(__name__)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Renderer(AbstractEngine):

    def build_model(self):
        LOGGER.info("* Creating Model.")
        model = instantiate(self.cfg.model).to(self.device)
        model.eval()
        return model, None

    def build_dataset(self):
        #TODO
        LOGGER.info("* Creating Dataset.")
        dataset = {
            split: instantiate( #TODO: CameraPose, replicate other data
                self.cfg.dataset,
                split=split,
                mode='render_path',
                render_view_number=self.cfg.render_view_number,
                num_rays=None,
                cache_n_repeat=None,
            )
            for split in self.cfg.eval_splits
        }
        meta_data = {
            split: dataset[split].build_pose_meta_info()
            for split in dataset.keys()
        }
        return dataset, meta_data

    def _preprocess(self, data, split):
        # to gpu
        for k, v in data.items():
            if k == "rays":
                data[k] = namedtuple_map(lambda x: x.to(self.device), v)
            elif isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                pass
        # update pose info for this frame
        meta_data = self.meta_data[split]
        idx = meta_data["meta_ids"].index(data["meta_id"])
        data["bones_rest"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_rest"]
        )
        data["bones_posed"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_posed"][idx]
        )
        if "pose_latent" in meta_data:
            data["pose_latent"] = meta_data["pose_latent"][idx].to(self.device)
        return data
    
    def run(self) -> float:  # noqa
        if self.init_step < 0 and (not os.path.exists(self.cfg.resume_dir)):
            LOGGER.warning(
                "Ckpt not loaded! Please check save_dir: %s or resume_dir: %s." % (
                    self.save_dir, self.cfg.resume_dir
                )
            )
            return 0.

        assert self.world_size==1, 'Only support render_path when GPU=1'
        device = "cuda:0" 
        os.makedirs(os.path.join(self.save_dir, self.cfg.eval_cache_dir,'image'), exist_ok=True)
        
        self.model.eval()
        model = self.model.module if hasattr(self.model, "module") else self.model
        for eval_split in self.cfg.eval_splits:
            dataset = self.dataset[eval_split]
            LOGGER.info(
                "* Evaluation on eval_split %s. Total %d" % (eval_split, len(dataset))
            )
            rgbs = []
            for i, data in enumerate(dataset):
                data = self._preprocess(data, eval_split)
                rays = data.pop("rays")    
                rgb, _, _, _ = render_image(
                    model=model,
                    rays=rays,
                    randomized=False,
                    normalize_disp=False,
                    chunk=self.cfg.test_chunk,
                    **data,
                )    
                rgbs.append(rgb.cpu().numpy())  
                image_path = os.path.join(self.save_dir, self.cfg.eval_cache_dir, 'image/{:03d}.png'.format(i))
                imageio.imwrite(image_path, to8b(rgbs[-1]))
                LOGGER.info('Save as '+image_path)
            rgbs = np.stack(rgbs, 0)
            video_path = os.path.join(self.save_dir, self.cfg.eval_cache_dir,f'{self.cfg.render_view_number}.mp4')
            imageio.mimwrite(video_path, to8b(rgbs), format='mp4', fps=10, quality=8)
            LOGGER.info("Save as "+video_path)

        return 1.0