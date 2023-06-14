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
https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L99-L104
* batch_size=1, fetch one frame by one camera at a time and generate a batch of rays.
https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/configs/dataset/zju.yaml#L1
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/zju_loader.py#L46
https://github.com/ChenYutongTHU/TAVA_NARF/blob/c3797b63b639d25a6bcabacd65c230e62c5ed508/tava/datasets/abstract.py#L7
The batch dim is squeezed by
https://github.com/ChenYutongTHU/TAVA_NARF/blob/06372e7ad738fd4b0c4161b3c0aa81acc7741106/tava/engines/trainer.py#L29C1-L30
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
    * 1. rgba \[H,W,4] $\leftarrow$ self.parser.load_image()^self.parser.load_mask() (^=concat) (In ZJU, both image width and parser.width are 1024 but the resize_factor is 0.5. So here H/W is 512)
    * 2. undistort *K,D*
    * 3. resize *resize_factor, default=0.5??*
    * 4. normalize to \[0,1]
    * 5. prepare camera *(K-intrinsic, resize_factor, extrinsic, self.parser.WIDTH-1024, self.parser.HEIGHT-1024)* return a dict
    * 6. **generate_rays(cameras, opencv_format=True, near=self.near, far=self.far)** near/far-df-null
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L53-L62
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L104-L112
      * directions are not normalized while viewdirs are normalized. The magnitude of directions equal to the corresponding point on the surface of depth=1
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L73-L88
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
      * Mip-NERF: treat each ray as a cone.
      * According to Mip-NeRF Sec3.1: We set r to the width of the pixel in world coordinates scaled by 2/sqrt(12) which yields a cone whose section on the image plane has a variance in x and y that matches the variance of the pixel's footprint.(In the context of digital imaging, a pixel's footprint refers to the physical size or area occupied by a single pixel on a display or image sensor. )
      * A point (conventional NeRG) -> A 3D Gaussian frustrum: we need to compute the expected PE of all points in the frustrum. It has closed form and we only need to compute the diagonal of the costy variance matrix. 
    * 8. Finalize rays generation
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/utils/camera.py#L104-L112
      Rays() is just a simple namedtuple
 #### preprocess_data()
* preprocess_data() return
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L83-L84
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L117-L122
  * fetch_data() generates the ray directions while preprocess_data() select valid rays and associates them with pixel labels
  * training: return rays' shape = \[H*\W, 3] (note that rays is still a named_tuple in which each item is shaped as \[H*\W,3])
* Step-by-step
  * 1. random background: 
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L86
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L90
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L100
    * The alpha channel is decided by mask. 0:background 1:foreground 0.5:ignore (fuzzy boundary).
    * Thus we should ignore places whose alpha is not 0 or 1.
    * In ZJU, the provided mask is uint8 (0 or 1), however tri_map=True pixels around the boundary is set to 0.5. So not all rays are valid.
  * 2. Select num_rays valid rays (alpha is not 0 or 1)
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L105-L108
  * 3. Tie them with pixel labels
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L109
  * 4. Reorder and reshape rays according to the selected indices
    https://github.com/ChenYutongTHU/TAVA_NARF/blob/2146465223c669875915e7a754ed7502f482e99f/tava/datasets/zju_loader.py#L110-L113
    * rays: origins, directions, viewdirs are shaped as (H*W, 3)
 
 ### Training - Optimization
 * Learning rate
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L48-L55
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/utils/training.py#L76-L78
   log-linear interpolation
 * Adam
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L66-L70
 
 ### Training - train_step
 https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L121
 #### 1. \_preprocess
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L253
   * .to(gpu)
   * add "bones_rest","bones_posed",("pose_latent") to data. (Read from self.meta_data using frame_id)
       * About meta_data
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L91-L94
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/datasets/zju_loader.py#L180
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/1f5bccf8e0e4f802efa826a248cafaa4ffc4d5eb/tava/datasets/zju_loader.py#L239-L244
         * As defined in zju_parser.py - 24 Joints including a root.
         * meta_ids = frame_ids
         * bones_rest: named_tuple(heads=None, tails<Location (3,)>, transform<(4,4) matrix>) in canonical space
         * bones_posed: named_tuple(heads=None, tails<Location (3,)>, transform<(4,4) matrix>) in deformed space
             * Heads=None means that the transform has already been producted as the tree's order? **YES!**
               * when computing the closest points to each bone, head is computed as
                  https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/utils/bone.py#L19
                 * The last column of the transform matrix represents the translation from origin to the head's position?
                 * now both heads and tails are world coordinates.
         * pose_latent? meta_data\["params"]? 24*3(Joint's axis-angle)+3+3(global rotation+translation)
 #### 2. **Forward**
   * 1. **Model:tava.models.mipnerf_dyn.DynMipNerfModel - Canonical space $(x,\Sigma)\rightarrow$(rgb, density, ambient occlusion)**
      ```
      model:
      _target_: tava.models.mipnerf_dyn.DynMipNerfModel
      pos_enc: ${pos_enc} # snarf
      shading_mode: "implicit_AO"
      shading_pose_dim: ${dataset.pose_latent_dim} # 78
      world_dist: ${dataset.world_dist} # 0.3 only consider rays whose distance to bones are smaller than 0.3
      ```
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/mipnerf_dyn.py#L87-L100
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/basic/mipnerf.py#L19-L28
      * Shading_mode: Posing leads to self-occlusion and thus affects shading
         * None: do not model pose-dependent shading
         * Implicit: pose-conditioned color (query rgb conditioned on both hidden features and pose)
         * Implicit_A0: (described in paper), output a scalar - ambient occlusion
         * Does the MLP not conditioned on View direction?
      * **self.pos_enc: tava.models.deform_posi_enc.snarf.SNARFDPEncoder - Root finding to map a query point on the deformed pose to candidate points on the canonical pose (world coordinate).** and compute return position embedding of canonical point
        ```
        _target_: tava.models.deform_posi_enc.snarf.SNARFDPEncoder
         posi_enc:
           _target_: tava.models.basic.posi_enc.IntegratedPositionalEncoder
         n_transforms: ${dataset.n_transforms} # 19
         offset_net_enabled: true
         offset_pose_dim: ${dataset.pose_latent_dim} # 78
         offset_constant_zero: false
         cano_dist: ${dataset.cano_dist} # 0.3
        ```
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L69-L70
        Instantiated by hydra
        * self.posi_enc: *IntegratedPositionalEncoder* position encoder of (x, sigma), designed for Mip-NeRF (cone rays), used to compute PE of canonical points
        * self.skin_net: an MLP predicts LBS weights $(x,y,z)\rightarrow (n_transforms+1)$. As mentioned in TAVA-Sec.3.3-ForwardSkinning, 'introducing an additional term $w_{bg}\*I_d$. This term allows the points in the background and empty space to not follow the skeleton when it is deformed'. However, in the code, ${w_{bg}}$ also depends on $x_c$ while in the paper $w_{bg}$ is point-independent. *PositionalEncoder*
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L17
        * self.offset_net: Pose-dependent NRigid deformation besides LBS $(x,y,z, P)\rightarrow \delta\in\mathbb{R}^3$
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L40 
        *PoseConditionDPEncoder* Concatenate PositionalEncoder(x,y,z) and pose latent ($\mathbb{R}^{78}$)
        * **Forward skinning** step-by-step
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L230
            * 1. query LBS weights including background weights - self.skin_net
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L254
            * 2. query Non-rigid offsets - self.offset_net
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L276
            * 3. Compute
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L280-L282
               
        * **Root-finding** step-by-step (How to train the root finding step? - inject gradient)
            * 1. Initialize - Sec.3.3 Inverse skinning. Use K=5 initialization. (no_grad)
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L171-L172
            * 2. Newton method (no_grad)
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L182
            Here, we need to compute the Jacobian (gradient of forward_skinning w.r.t x_c) (enable_grad = True)
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L410-L411
            Iterative optimization 
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L629-L633
            Note that the optimization is applied on batches of points. For some points that already converge, we need to stop the optimization for then and mask them.
            * 3. Filter invalid canonical corresponding points 
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/models/deform_posi_enc/snarf.py#L192-L199
            If canonical points are >0.3 from the nearest bones, then take them.
            * 4. **Inject gradient!!** Note that we need to inject gradient to the canonical correspondence point otherwise the skin_net and offset_net cannot be trained. See Equation(9) in the paper. Here the code use some trick to inject gradient without changing the value of x_c.
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/deform_posi_enc/snarf.py#L204-L205
        * The SNARF model also needs to embed the position encoding. (**It's not done in the DynNeRF!!**)
            https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/deform_posi_enc/snarf.py#L220
        
        
   * 2. **Forward**
      * 1. Some regularization for skin_net (LBS) and offset_net (Non-rigid deformation)
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/mipnerf_dyn.py#L246-L254
         * 1. Skinning weights on the bones should be one-hot.
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/86aa45b40239023ac7949cbf4511d60fe521c88c/tava/models/deform_posi_enc/snarf.py#L470
         * 2. Offsets on the bones should be all zero, including the joints.
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/86aa45b40239023ac7949cbf4511d60fe521c88c/tava/models/deform_posi_enc/snarf.py#LL481C6-L481C6
      * 2. Calculate near and far, select rays  (prevent sampling too many empty points)
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/mipnerf_dyn.py#L258-L260
         * The closest distance between two segments. (Very convoluted, there might be some elegant understanding utilizing linear algebra).
         * Only select rays whose distances to the nearest bones are smalled than self.world_size (0.3).
         * For each selected ray, Near/Far is computed according to the closest point on the rays to the bones. (+- t_margin)
      * 3. stratified query (df num_levels=2)
         * level_i=0 :sample_along_rays
            * linearly split \[near,far]
            * randomized? randomly select a point in each segment: select the endpoint of each segment
            * generate (means, covs) for mipNeRF (cast_rays)
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/basic/mipnerf.py#L345
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/basic/mipnerf.py#L242-L243
         * level_i=1: resample_along_rays (grads are not propagated across levels)
            * An additional param: weight  
            * Sample from inverse CDF （Somehow complicated, please re-check it）
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/basic/mipnerf.py#L386-L391
         * query_mlp: note that now samples are (mean, variance) of frustum cones.
          https://github.com/ChenYutongTHU/TAVA_NARF/blob/01449cd2417c5abbcd17aab2937686d1525acb87/tava/models/basic/mipnerf.py#L127-L130
            * a. Use SNARFEncoder to find canonical correspondence (IPE embedded):
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L177-L184
               * x_enc: encoded embedding of x_cano = self.posi_enc(x_cano, x_cov). Note that when transform x_world to x_cano, we don't transform the x_cov.
               * x_warp: coordinate of x_cano
               * mask: is_convergence after Newton method **and** is the yielded canonical point in the range (closest distance to the bone < cano_dist0.3)
               * valid (also a mask): is_convergence after Newton method
            * b. Query the rgb, density and ambient occlusion of the canonical points.
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L198-L200
               for canonical points whose mask=False, the returned feature/sigma=
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L211-L216
               rgb_padding? default=0.001 rgb+-\[-1,1]\*0.001
            * c. Combine density and rgb (Now we need to use mask to exclude invalid roots)
              https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L218-L232
            * d. Final return
              https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L233
              Note that now the x_warp is the canonical point with the largest density and valid indicates whether each query world point finds corresponding canonical point. (If for the ith world point, all corresponding canonical points diverge, then mask\[i]=False).
       * 4. interpolate along rays?
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L336-L338
         For query points on one ray, there might be some points which we cannot find valid canonical correpondences and hence cannot query its rgb, density and ao. Thus we need to interpolate its (rgb, density and ao.) by its neighboring valid points.
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/engines/trainer.py#L258
       * 5. Force Out-of-Range points (closest distance to bone > world_dist=0.3) to density=zero
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L341-L347
       * 6. Volumn rendering
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/7399d1934a06986a2db295d909c02246e4f4e9c1/tava/models/mipnerf_dyn.py#L350-L357
         * Calculate alpha and transparancy.
         * Weights = alpha\*transparancy (will be used later level_1 sampling).
       * 7. Final output
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/b6a9a771dcd2de32961bf4420ceba8ff6aa6cfcd/tava/models/basic/mipnerf.py#L138-L140
         * each level of output is restored and filled.
          https://github.com/ChenYutongTHU/TAVA_NARF/blob/b6a9a771dcd2de32961bf4420ceba8ff6aa6cfcd/tava/models/mipnerf_dyn.py#L362
          Only for selector=True, rgb is stored as the output, otherwise as the color_bkgd.
         * extra_info=None
         * ret=[(comp_rgb, disp, acc, points)]*n_level
            * comp_rgb shape (N,3)
            * disp inverse to depth (N,)
            * acc ?(N, weights.sum(-1) (is background or foreground
               https://github.com/ChenYutongTHU/TAVA_NARF/blob/b6a9a771dcd2de32961bf4420ceba8ff6aa6cfcd/tava/models/basic/mipnerf.py#L295
            * weights = alpha\*transparency
 * 2. **Loss computation and Backward**
   https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/configs/mipnerf_dyn.yaml#L53-L56
    * 1. RGB loss (Coarse and fine)
       https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/engines/trainer.py#L267
       https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/engines/trainer.py#L273
    * 2. regularization loss (weight=0) 
    * 3. Backward
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/engines/trainer.py#L298-L299
       
   
         
        
              
               
         
               
         











