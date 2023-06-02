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
        |   ├── 'f': faces
        |   ├── 'shapedirs': shape blend shape
        |   ├── 'posedirs':  Pose blend shape (6890*3, 207) \#207=23*9
        |   ├── 'J_regressor': ? numpy [J, n_verts] # Regress joint locations from vertices. (after adding shapedirs)
        |   ├── 'v_template': [batch_size, n_verts, 3]
        |   ├── 'kintree_table: parents [J,] 
        |   └── 'weights': [n_verts, J + 1] 
        'posedirs', 'shapedirs'
        ├── CoreView_313/
        |   ├── annots.npy #meta_data dict
        |       ├── 'ims': [frame0, frame1, ...]
        |       └──?
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
      * Forwarding (pose_72, shape_10, global_transformation) to SMPL, we can compute (vertices, joints, joints_transform, bones_transform). Note that we have 23+1(root) joints and 23 bones. Each bone's transformation is represented by a 3-d vector? (Rotation+Translation?)
        https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/body_model.py#L99-L109
        Pose param is 
        * **axis–angle representation** （axis, angle). Here the axis-angle representation is a 3d vector whose direction represents the axis and norm represents the counter-clockwise angle (pi).
        * We then transform it into a quaternion $[cos\frac{\theta}{2}, w\times sin(\frac{\theta}{2})]$
        * The quaternion is transformed into a matrix. $3\times3$ Thus each pose blendshape has 3\*3=9 params.
   * 2. Load meta data from data/zju/CoreView_313/annots.py
   * 3. Load rest post info
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L84-L87
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/e94e6571875eaa04b248163e266a665c3a74ea87/tools/process_zju/main.py#L19-L26
      For rest post, pose_params=R=T=zero
      
  
