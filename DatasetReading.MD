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
        ├── CoreView_313/
        |   ├── annots.npy #meta_data dict
        |       ├── 'ims': [frame0(dict), frame1(dict), ...] 
                     for each frame:
                     ├── 'ims': ['Camera_B1/000000.jpg', 'Camera_B2/000000.jpg',... , 'Camera_B23/000000.jpg'] (23 views)
                     └── 'kpts2d': (23,25,3)  (view_id, kpt_id, location)
        |       └── 'cams': {'K': [3*3,3*3,...,3*3], 'D': [5*1], 'R': [3*3],'T': [3*1]} (23 views)
        |   ├── new_params
        |       ├── 1.npy #save smpl data 
        |           ├── 'shapes': 10-dim         
        |       └──? 
        ├── ...
        └── CoreView_386/
  ```
* Preprocess the downloaded data,  for each subject id
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
   * 3. Load rest post info
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L84-L87
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L19-L26
      For rest post, pose_params=zero_vector s.t. the 3\*3 derived rotation is an identity matrix; 
      Note that the pose_params only defines rotation relative to its parent. Identity relative rotation-matrix refers to the rest pose.  To compute the transformed joint position, we need to go through the kinematic tree.
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/lbs.py#L106-L118
      * J_transformed: the location of posed joints. (Note the the 3D coordinates are the last column of the joint transformation matrix, overall translation from the root joint)
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/lbs.py#L143-L144
        * **A/rel_transforms**: **relative to** the rest joints, the rigid transformation (R,T) of the posed joints. Note that here R is the same as the cumprod transformation matrix's R but the T is the translation between posed joints and rest joints. This transform is linearly weighted in LBS.
        * A_bone: the transformation (relative-to-root) of the bone's head joint.
      
  
