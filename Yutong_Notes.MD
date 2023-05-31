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
      * Mip-NERF: treat each ray as a cone.
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
             * Heads=None means that the transform has already been producted as the tree's order?
               * when computing the closest points to each bone, head is computed as
                  https://github.com/ChenYutongTHU/TAVA_NARF/blob/c709f985b4a89ec7077999a81db2f8fa702c4062/tava/utils/bone.py#L19
                 * The last column of the transform matrix represents the translation from origin to the head's position?
                 * now both heads and tails are world coordinates.
         * pose_latent? meta_data\["params"]?
 #### 2. **Forward**
   * 1. **Model:tava.models.mipnerf_dyn.DynMipNerfModel - Canonical space $(x,\Sigma)\rightarrow$(rgb, density, ambient occlusion)**
      ```
      model:
      _target_: tava.models.mipnerf_dyn.DynMipNerfModel
      pos_enc: ${pos_enc} # snarf
      shading_mode: "implicit_AO"
      shading_pose_dim: ${dataset.pose_latent_dim} # 78
      world_dist: ${dataset.world_dist} # 0.3
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
        * self.skin_net: an MLP predicts LBS weights $(x,y,z)\rightarrow (n_transforms+1)$. As mentioned in TAVA-Sec.3.3-ForwardSkinning, 'introducing an additional term $w_{bg}\*I_d$. This term allows the points in the background and empty space to not follow the skeleton when it is deformed'. However, in the code, ${w_{bg}} also depends on $x_c$ while in the paper $w_{bg}$ is point-independent. *PositionalEncoder*
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
      * 1. for SNARFDPEncoder
         https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tava/models/mipnerf_dyn.py#L246-L254
         Note t
      * 2. 
         











