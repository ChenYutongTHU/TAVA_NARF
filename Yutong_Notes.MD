# High-level structure

https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/launch.py#L64-L65

* engine: trainer/evaluator

## Trainer
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L33-L34
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L97-L98

### Abstract
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L13-L14
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L35
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/abstract.py#L57
* build_model(), build_dataset() are defined by Trainer()

### DataLoader
* Abstract loader: [abstract.py](tava/datasets/abstract.py)
* Subject loader: [zju_loader.py](tava/datasets/zju_loader.py)
* Parser: build a parser for a single person (*subject_id*).
    * Save WIDTH, HEIGHT 1024
    * Save camera pose (in+ex-trinsics) as *self.camera*  (K-intrinsics,D-distortion,w2c_4$\times$4)
    * Save data_dir, image_files (\[f1\[c1,c2], f2\[c1,c2]])
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_parser.py#L118-L122
    * Define load_image(), load_mask() 
https://github.com/ChenYutongTHU/TAVA_NARF/blob/9ba5a227cb9213ca6993929ad902e175f5cf4aab/tava/engines/trainer.py#L77

https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/configs/dataset/zju.yaml#L1
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/zju_loader.py#L46
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/abstract.py#L7
* **abstract.py**/CachedIterDataset also define \__getitem__()
```
data = self.fetch_data(index)
...
return self.preprocess(data)
```
* fetch_data() and preprocess() are defined by SubjectLoader() in **zju_loader.py**
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/zju_loader.py#L124-L127
#### fetch_data()
* fetch_data() returns
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L168-L178
* ZJU is a single-person multi-view video dataset. Each training split has a single *subject_id*. Each sample is tied with *(frame_id, camera_id)*.
* Step-by-step - **Goal: associate Rays(r,d) and its pixel label**
    * 1. rgba \[H,W,4] $\leftarrow$ self.parser.load_image()^self.parser.load_mask() (^=concat)
    * 2. undistort *K,D*
    * 3. resize *resize_factor, default=0.5??*
    * 4. normalize to \[0,1]
    * 5. prepare camera *(K-intrinsic, resize_factor, extrinsic, self.parser.WIDTH-1024, self.parser.HEIGHT-1024)* return a dict
    * 6. **generate_rays(cameras, opencv_format=True, near=self.near, far=self.far)** near/far-df-null
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L53-L62
* **Transform image pixel to ray points**
    * 1. Distort and resize 2D images (D is not included in the Camera())
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L142-L153
    * 2. Prepare Camera (Exclude distortion and include a resize_factor)
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L156-L163
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L39-L43
      Now *K* becomes $[[0.5f_x,0,0.5][0,0.5f_y,0.5][0,0,1]]$
      *resize_factor=0.5* is equal to moving the image plane closer and scaling the image by $1/2$.
    * 3. Generate (u,v) grid
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L67-L71
    * 4. Map (u,v) to ray directions  (camera coordinate)
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L73-L81
      **?? +0.5**
    * 5. Rotate (translation is not needed) ray directions (camera coor) to ray directions in the world coor
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L86-L87
      Now the shape of directions is ((n_camera),H,W,3) (Note that in ZJU just H,W,3). Please note the equivalence between matrix-product and (hadamard-product+reduce-sum)
    * 6. Normalize ray directions
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L88
    * 7. Compute radius?
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L91-L98
    * 8. Finalize rays generation
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L104-L112
      Rays() is just a simple namedtuple
 #### preprocess_data()
* preprocess_data() return
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L83-L84
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L117-L122
  * fetch_data() generates the ray directions while preprocess_data() select valid rays and associates them with pixel labels
  * training: return rays' shape = \[H*W, 3] (note that rays is still a named_tuple in which each item is shaped as \[H*W,3])
* Step-by-step
  * 1. random background: 
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L86
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L90
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L100
    * The alpha channel is decided by mask. 0:background 1:foreground 0.5:ignore (fuzzy boundary).
    * Thus we should ignore places whose alpha is not 0 or 1.
  * 2. Select num_rays valid rays (alpha is not 0 or 1)
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L105-L108
  * 3. Tie them with pixel labels
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L109
  * 4. Reorder and reshape rays according to the selected indices
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L110-L113
    * rays: origins, directions, viewdirs are shaped as (H*W, 3)
    











