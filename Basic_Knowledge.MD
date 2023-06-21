# Fundamental knowledge
## Human rigid transformation with kinematic tree
### First of all, keep in mind what these parameters represent
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Notations</th>
      <th>Describe</th>
      <th>Representation</th>
      <th>Defined in</th>
      <th>misc</th>
    </tr>
  </thead>
  <tbody>
     <tr>
      <th>Joint positions </th>
      <th>$\mathbf{J_i}\in \mathbb{R}^3$</th>
      <th>Skeleton, bone's length</th>
      <th>(x,y,z)</th>
      <th>$\in$ Shape and joint regressor </th>
      <th></th>
    </tr>
     <tr>
      <th>Joint's SO(3) </th>
      <th>$\mathbf{w}\in SO(3)$</th>
      <th>Non-rigid deformation</th>
      <th>axis-angle 3-dim</th>
      <th>$\cong$ SMPL Pose params</th>
      <th>-</th>
    </tr>
    <tr>
      <th>Joint's world transformation</th>
      <th>$\mathbf{G}\in \mathbb{R}^{3\times 4}$</th>
      <th>relation between world the joint's local coordinate</th>
      <th>homo([SO(e)|T])</th>
      <th>$\cong$ Kinematic tree+Joint's SO(3)+skeleton-xyz</th>
      <th>-</th>
    </tr>
     <tr>
      <th>Joint's relative transformation (Joints_transform or rel_transform) </th>
      <th>$\mathbf{G'}\in \mathbb{R}^{3\times 4}$</th>
      <th>relation between the coordinate system of the joint in the rest and deformed pose (Including global transformation)</th>
       <th>homo([SO(e)|T])</th>
      <th>$\cong$ $G'$ </th>
      <th>-</th>
    </tr>
     <tr>
      <th>Bones transformation</th>
      <th>$\mathbf{G'}\in \mathbb{R}^{3\times 4}$</th>
      <th>How points on the bone transform = head joint's world transform</th>
       <th>homo([SO(e)|T])</th>
      <th>$\cong$ $G$ </th>
      <th>-</th>
    </tr>    
  </tbody>
</table>

### Next, be careful that ...
* The root joint's SO(3) is always axis-angle (0,0,0) because it is locked and cannot move physically. The root's world coordinate is **not** (0,0,0).
* We define global transform (Rh and Th). After multiplying the global transformation, the bone transformation whose head is the root joint and joints transform of the root are **no longer** I. (In fact, the joints_transform equals to the global_transform).\
* Difference between LBS in data processing (lbs.py) and TAVA:
    * In lbs.py, we use the 24 joint's relative transform (Joints_transform). For example, for the last joint - rhand, its transformation will cause how points on the bone whose head is rhand move.
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/529f266c2caf61c341836ba82ccb404a59295f48/tools/process_zju/lbs.py#L43-L60
    * However in TAVA, we only use (19+1-bg) joint's relative transform and the joint's transform are re-calcutated by bones_transform.
      https://github.com/ChenYutongTHU/TAVA_NARF/blob/529f266c2caf61c341836ba82ccb404a59295f48/tava/models/deform_posi_enc/snarf.py#L259-L264

      https://github.com/ChenYutongTHU/TAVA_NARF/blob/529f266c2caf61c341836ba82ccb404a59295f48/tava/models/deform_posi_enc/snarf.py#L282-L284

      This means that 24-19=5 joint's transform are missing. Particularly, the 19+1 weighted transform ignores the 5 bones whose heads are ltoes, rtoes, head, lhand and rhand.



## Evaluation metric
### PSNR
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/eaaa2f3036e1d33453ffa1526400ea6d6cdfcac3/tava/utils/training.py#L108-L117
### SSIM
  
  

 
