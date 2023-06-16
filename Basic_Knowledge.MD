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
      <th>(x,y,z)</th>
      <th>$\cong$ SMPL Pose params</th>
      <th>-</th>
    </tr>
     <tr>
      <th>Joint's head->tail transformation </th>
      <th>$\mathbf{G}\in \mathbb{R}^{4\times 4}$</th>
      <th>relation between the tail's and head's coordinate</th>
      <th>homo([SO(e)|T])</th>
      <th>$\cong$ SMPL Pose params+Joint positions</th>
      <th>-</th>
    </tr>
    <tr>
      <th>Joint's origin->tail transformation </th>
      <th>$\mathbf{G'}\in \mathbb{R}^{3\times 4}$</th>
      <th>relation between world's and the tail's coordinate</th>
       <th>homo([SO(e)|T])</th>
      <th>$\cong$ Kinematic tree+head->tail tf</th>
      <th>-</th>
    </tr>
     <tr>
      <th>Joint's rest->pose transformation </th>
      <th>$\mathbf{Gr}\in \mathbb{R}^{3\times 4}$</th>
      <th>relation between the coordinate system of the joint in the rest and deformed pose</th>
       <th>homo([SO(e)|T])</th>
      <th>$\cong$ $G'$ </th>
      <th>-</th>
    </tr>
    
  </tbody>
</table>

### Next, be careful that ...
* **Joint's SO(3): ** - 
  Bones (segments) vs Joints (points) are different. A bone consists of two joints.
  * Each joint is defined by two factors:
    * (a) its 3D coordinates (x,y,z), either defined in the global coordinate system (global-(x,y,z))  or relative to its parent joint (local-(x,y,z))
    * (b) its rotation, axis-angle /SO(3)[special orthogonal group](https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group). Although 3D-(x,y,z) already defines the joint's location, it **cannot** define where to put its children bones. Note that we do not say SO(3) is defined in the global or local coordinate system. The ball joint just rotates around the center of itself.
  * Each bone consists of two joints but we **do not need** to define two sets of (a)(b) to fully describe the **pose** of a bone whose length is given. This is because the bone has a fixed length and cannot be blended (rigid). (The length is already defined in the rest pose.) We can first decide the global-(x,y,z) of the head joint and the rest is to decide the global/local-(x,y,z) of the tail joint. Again, given the length is fixed, we can only rotate bone around the head joint and thus the end-point position can be fully described by the SO(3) of the head joint. 
  * As a result, on a single human-skeleton with bones' length pre-defined in the rest pose, for each bone, the head joint's SO(3) decides where we should put the tail joint. 
  * Although the tail joint's SO(3) has nothing to do with the current bone, it will affect the tail joint's children.
  * Therefore, the following two terms are basically the same thing. They can be represented as 3-dim angle-axis (used as SMPL-pose), quaternion or rotation matrix (used in transformation computation).
    *  (i) the head joint's SO(3)
    *  (ii) How points on the bone transform **relative to** the head joint when the head joint rotates from the rest pose (rotation matrix=identity) to the current pose.
    (i) is the cause while (ii) is the result.  Why we need (i)\_\SO(3) rather than (ii)-(x,y,z) to represent the bone? Because SO(3) can describe how the movement of the head joint can affect all its children's joints? (pass along the kinematic tree).
   * Somehow confusing here, each joint may be included in multiple bones. It can have different SO(3) in multiple bones. Thus, just to be clear, we would say SO(3)/angle-axis/rotation matrix is a property of a bone rather than a joint.
* **Bone's tail-head coordinate transformation**: $G\in \mathbb{R}^{3\times4}$ = \[R|T\] - tail-relative coordinate system $\rightarrow$ head-relative coordinate system.
  * The bone's rotation matrix $R_i$ and the local-(x,y,z) of the tail joint (relative to its head joint), which can also be seen as the tail's translation to the head, forms the transformation of this bone relative to its parent. We denote it by $G_i\in \mathbb{R}^{3\times4}$.
  * Each joint (SO(3)) corresponds to a coordinate system. $G$ projects **the tail's coordinate system to the head's coordinate system.** For instance, the origin's coordinate in the tail's coordinate system is $x_l = (0,0,0)$, applying $G$ to $x_l$ obtains (t_x,t_y,t_z) -- its coordinate in the head's coordinate system.
  * Another approach to understanding the transformation is to imagine two x-y-z cube axes. The first cube is the head's coordinate while the second cube is the tail's coordinate. How can we non-rigidly transform the first cube to the second cube. (Rotate and translate)
* **Kinematic tree.**  -> Bone's tail-origin coordinate transformation
  * For a joint, we trace along its ancestors util the root and apply $G$ one by one, we can get an accumulated transformation. Note that the parent of the root is the world origin. And thus the root head-tail transformation is a transformation deviated from the world coordinate. The accumulated transformation describes how coordinates in the bone's tail coordinate system are transformed to the world coordinate.
* Combining all the ingredients: Non-rigid transformation from the rest pose to the current pose.
  * For the rest pose, we define the SO(3) of all joints as the identity (rotation matrix) or zeros (anxis-angle). So the $G'$ of all joints are \[I|T] where T is the global coordinate of the tail joint. 
  * Thus,  G'(pose)\dotproduct G'(rest)\_inv transforms the rest pose's joint coordinate  (world->pose) \dotproduct (world<-rest) = (rest->pose). Denote as Gr(pose) Again, two approach to understanding this:
    *  Gr defines how to non-rigidly transform the joint's x-y-z cube in the rest pose to its x-y-z cube in the current pose.
    *  How can we transform a point in the posed coordinate system into its rest coordinate system.
  * This can be easily formulated as

## Evaluation metric
### PSNR
  https://github.com/ChenYutongTHU/TAVA_NARF/blob/eaaa2f3036e1d33453ffa1526400ea6d6cdfcac3/tava/utils/training.py#L108-L117
### SSIM
  
  

 
