# ZJU_MoCap
## A. Dataset Structure
* 6 subject_ids \[313, 315, 377, 386, 387, 390, 392, 393, 394]
  ```
    data/
    ├── animal/
    |   ├── Hare_male_full_RM/
    |   └── Wolf_cub_full_RM_2/
    └── zju/
        ├── SMPL_NEUTRAL.pkl
        |   ├── 'f': faces (13776,3) [[1,2,0],] faces of indices
        |   ├── 'v_template': [6890, 3] -1.2~0.88
        |   ├── 'shapedirs': shape blend shape (6890, 3, 10) range -0.072~+0.05, magnitude 1e-2 !!replacement to v_template
        |   ├── 'posedirs':  Pose blend shape (6890*3, 207-9*J) \#207=23*9 (magnitude ~1e-4) !!replacement to shape
        |   ├── 'J_regressor': ? numpy [J+1, n_verts] # Regress joint locations from vertices. (after adding shapedirs), very sparse! 6~27 out of 6890 non-zero vertices for a single joint.
        |   ├── 'kintree_table: parents [J+1,] 
        |   └── 'weights': [n_verts, J + 1]  very sparse! e.g. 4 out of 24 non-zero joints for v1.
        ├── CoreView_377/
        |   ├── annots.npy #meta_data dict
        |       ├── 'ims': [frame0(dict), frame1(dict), ...] 
                     for each frame:
                     ├── 'ims': ['Camera_B1/000000.jpg', 'Camera_B2/000000.jpg',... , 'Camera_B23/000000.jpg'] (23 views)
                     └── 'kpts2d': (23,25,3)  (view_id, kpt_id, location)
        |       └── 'cams': {'K': [3*3,3*3,...,3*3], 'D': [5*1], 'R': [3*3],'T': [3*1]} (23 views)
        |   ├── new_params - estimated SMPL params for each frame
        |       ├── 0.npy #save smpl data 
        |           ├── 'shapes': 10-dim         
                ├──  1.npy **as rest pose!! we only use shape and force pose as all_zeros**
                    ├── 'poses': (1,72) 72=24*3 [0,:3]=[0,0,0] the axis-angle of the root joint is all-zero.
                    ├── 'shapes': (1,10) -0.1~+0.3
                    ├── 'Rh': (1,3) global rotation? axis-angle
                    └── 'Th': (1,3)
                ├── ...
                └── 616.npy
                    
        ├── ...
        └── CoreView_386/
  ```
* Preprocess the downloaded data,  for each subject id **Convert estimated SMPL-beta/pose params into vertices-3D coordinates and rest->pose bone transformation matrices**
   * 1. Load SMPL model -  data/zju/SMPL_NEUTRAL.pkl
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L72-L74
      * a nn.Module.
      * Forwarding (pose_72, shape_10, global_transformation) to SMPL, we can compute (vertices, joints, joints_transform, bones_transform). Note that we have 23+1(root) joints and 23 bones. Each bone's transformation is represented by a 3-d vector (axis-angle)
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/body_model.py#L99-L109
        Pose param is 
        * **axis–angle representation** （axis, angle). Here the axis-angle representation is a 3d vector whose direction represents the axis and norm represents the counter-clockwise angle (pi).
        * We then transform it into a quaternion $[cos\frac{\theta}{2}, \mathbf{w}\times sin(\frac{\theta}{2})]$
        * The quaternion is transformed into a matrix. $3\times3$ Thus each pose blendshape has 3\*3=9 params. See [Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived%20rotation%20matrix)
        * Note that for axis-angle = zero_vector (rest_post), the quaternion is \[1,0,0,0] and the derived rotation matrix is an identity matrix.
   * 2. Load meta data from data/zju/CoreView_313/annots.py, including path/to/each view-frame image, camera params of 23 views, keypoint 2D location. 
   * 3. Load rest pose info -- load 1.npy but we only use the shape estimation and set pose to all_zeros
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L84-L87
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L19-L26
      For rest post, pose_params=zero_vector s.t. the 3\*3 derived rotation is an identity matrix; 
      Note that the pose_params only defines rotation relative to its parent. Identity relative rotation-matrix refers to the rest pose.  To compute the transformed joint position, we need to go through the kinematic tree.
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/lbs.py#L106-L118
      * J_transformed: the location of posed joints. (Note the the 3D coordinates are the last column of the joint transformation matrix, overall translation from the root joint)
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/lbs.py#L143-L144
        * **A/rel_transforms**: **relative to** the rest joints, the rigid transformation (R,T) of the posed joints. Note that here R is the same as the cumprod transformation matrix's R but the T is the translation between posed joints and rest joints. This transform is linearly weighted in LBS. (In load_rest_pose, the rotation part should all be identities and the translation part is the 3D location relative to the head)
        * A_bone: the transformation (relative-to-**world_origin**) of the bone's head joint.
   * 4. Load pose info for each frame
       https://github.com/ChenYutongTHU/TAVA_NARF/blob/87426690eb48698d623c874697b0c6958b3bf05a/tools/process_zju/main.py#L91-L94
       after stacking all the frames, return data (dict)
       * "lbs_weights" : (6890,24) (from SMPL)
       * "rest_verts": (6890, 3) (from SMPL-template + estimated shape)
       * "rest_joints": (24, 3) (from SMPL-template + estimated shape)
       * "rest_tfs": (24, 4, 4) (rest_joints, rotation:identity, translation: tail-head location)
       * "rest_tfs_bone": (24, 4, 4) (~rest-tfs, yet the tfs of the head bone)
       * "verts": (Nframes, 6890, 3) (from rest_verts + estimated pose (including global r-t))
       * "joints": (Nframes, 24, 3) (from rest_joints + estimated pose (including global r-t))
       * "tfs": (Nframes, 24, 4, 4) (from rest_tfs + estimated pose (including global r-t))
       * "tf_bones": (Nframes, 24, 4, 4)
       * "params": (Nframes, 24*3+3+3) estimated pose (including global r-t)
       
   * 5. save as data/zju/CoreView_377/pose_data.pt **This is going to be used as meta_data in zju_parser.load_meta_data()**
       https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/engines/trainer.py#L241-L248
       Note that after reading meta_data from pose_data.pt, the key-value are then processed and reordered in 
       https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/datasets/zju_loader.py#L180
       Save as 
       * bones_rest and bones_posed(=tf_bones): include location of the tail joints (23,3) and the head joints' transformation (23,4,4). A transformation describing  World coordinate -> local coordinate
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/datasets/zju_loader.py#LL200C43-L200C43

        https://github.com/ChenYutongTHU/TAVA_NARF/blob/a0e2eabd514fa5e750ceec86eb31621682539555/tava/datasets/zju_loader.py#L208-L215
        Later on, the LBS is calculated as 
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/529f266c2caf61c341836ba82ccb404a59295f48/tava/models/deform_posi_enc/snarf.py#L259-L264
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/529f266c2caf61c341836ba82ccb404a59295f48/tava/models/deform_posi_enc/snarf.py#L282-L284

       * pose_latent = params (N, 78) 78 = 3*24 (SMPL-pose params, axis-angle rotation of each bone, (0,0,0) for root) + 3 (global rotation) + 3 (global translation)

## B. Scale
### SMPL
* V_template coordinate range
    * x [-0.87, 0.87] 1.6 (arm stretch length, width)
    * y [-1.16, 0.56] 1.72 height?
    * z [-0.12, 0.17] 0.4 (depth) (belly)
* Bounding box 
    <----x----O
           /  |
          /   |
         /    y
        z     |
       /      |
      /       ⬇️ 
    * right: rhand (22) x=-0.77
    * left: lhand (23) x=+0.77
    * right: rhand (22) x=-0.77
    * down:  ltoes (10) y=-1.14
    * up: head(15) y=+0.35
    * back: rhand (22) z=-0.05
    * front: rtoes (11)  z=0.09
* However, for SMPL, we only care about the **transformation(translation) scale** because we're going to reset the center later in our NeRF. Just remember that the body's height is around 1.7, width~1.6 (two arms stretched), depth~0.4.
* The **transformation** is also going to be multipled by a estimated global transformation (Rh, Th).
    * Before global transformation, the origin is around the root/belly.
    * However in ZJU, the global transformation is very weird. First, the rotation approximately switches z axis and y axis. So after global transformation. it's z rather than y aligned with the body's height. (Image someone in CT)
    * Thus, we see multi-camera's origins are circled in x-y plane and also have same z values. This is because now z is aligned with body's height.
    * In a nutshell, in SMPL's template, body's height is aligned with y but in ZJU's capture system. body's height is aligned with z and cameras are circled in x-y plane.


       
      
      
  
