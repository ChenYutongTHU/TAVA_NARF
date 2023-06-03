# Fundamental knowledge
## Human rigid transformation with kinematic tree
### First of all, keep in mind what these parameters represent
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Notations</th>
      <th>Used as</th>
      <th>Representation</th>
      <th>It defines ..</th>
      <th>misc</th>
    </tr>
  </thead>
  <tbody>
     <tr>
      <th>Joint positions </th>
      <th>$\mathbf{J_i}\in \mathbb{R}^3$</th>
      <th>Translation terms in </th>
      <th>(x,y,z)</th>
      <th>The rest pose and **translation items** in </th>
      <th></th>
    </tr>
     <tr>
      <th>Parent-relative Rotation</th>
      <th>$\mathbf{W_i}\in \mathbb{R}^3$</th>
      <th>SMPL Pose parameters</th>
      <th>Axis-angle</th>
      <th>The rest pose and **translation items** in </th>
      <th>-</th>
    </tr>
  </tbody>
</table>
### Next, be careful that ...
* Bones (segments) vs Joints (points) are different. 
  * A bone consists of two joints.
  * Each joint is defined by two factors:
    * (a) its 3D coordinates (x,y,z), either defined in the global coordinate system (global-(x,y,z))  or relative to its parent joint (local-(x,y,z))
    * (b) its rotation, axis-angle /SO(3)[special orthogonal group](https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group). Although 3D-(x,y,z) already defines the joint's location, it **cannot** define where to put its children bones. Note that we do not say SO(3) is defined in the global or local coordinate system. The ball joint just rotates around the center of itself.
  * Each bone consists of two joints but we **do not need** to define two sets of (a)(b) to fully describe the **pose** of a bone whose length is given. This is because the bone has a fixed length and cannot be blended (rigid). (The length is already defined in the rest pose.) We can first decide the global-(x,y,z) of the head joint and the rest is to decide the global/local-(x,y,z) of the tail joint. Again, given the length is fixed, we can only rotate bone around the head joint and thus the end-point position can be fully described by the SO(3) of the head joint. 
  * As a result, on a single human-skeleton with bones' length pre-defined in the rest pose, for each bone, the head joint's SO(3) decides where we should put the tail joint. 
  * Although the tail joint's SO(3) has nothing to do with the current bone, it will affect the tail joint's children.
  * Therefore, the following two terms are basically the same thing. They can be represented as 3-dim angle-axis (used as SMPL-pose), quaternion or rotation matrix (used in transformation computation).
    *  (i) the head joint's SO(3)
    *  (ii) How points on the bone transform **relative to** the head joint when the head joint rotates from the rest pose (rotation matrix=identity) to the current pose.
   * Somehow confusing here, each joint may be included in multiple bones. It can have different SO(3) in multiple bones. Thus, just to be clear, we would say SO(3)/angle-axis/rotation matrix is a property of a bone rather than a joint.
* The bone's rotation matrix $R_i$ and the local-(x,y,z) of the tail joint (relative to its head joint), which can also be seen as the tail's translation to the head, forms the transformation of this bone relative to its parent. We denote it by $G_i\in \mathbb{R}^{3\times4}$.

 
