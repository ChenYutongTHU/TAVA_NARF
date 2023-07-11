# Forward/Backward Deformation
## Denotation
All coordinates and transformation matrices are homogeneous. We assume that all objects are globally registered, so we don't consider global rotation and translation.
* Deformed/Posed coordinates $x_p\in\mathbb{R}^4$
* Rest coordinates (canonical pose) $x_c\in\mathbb{R}^4$
* Pose parameters: $\theta\in\mathbb{R}^{3\times K}$, axis-angle
* Number of bones: $K$
* Transformation matrix of canonical pose: $\{B_c^{k}\in\mathbb{R}^{4\times4},k=1,\cdots,K \}$. 
* Transformation matrix of deformed pose (decided by $\theta$): $\{B_p^{k}\in\mathbb{R}^{4\times4},k=1,\cdots,K \}$
* Transformation from canonical pose to deformed pose: $B^{k}_{c\rightarrow p} = B_p^k{B_c^k}^{-1}\in\mathbb{R}^{4\times4}$
* Transformation from deformed pose to canonical pose: $B^{k}_{p\rightarrow c} = B_c^k{B_p^k}^{-1}\in\mathbb{R}^{4\times4}$
* One identity matrix $I$ for the background. That is $B_{c\rightarrow p}^{K+1}=I$.
* Number of bones: $K$
* How about background?

## Skinning
Each $x_c$ has a weight vector $W_c(x_c)\in \mathbb{R}^{K+1}$. The last one for the background.

### Forward skinning
$\mathcal{S}_{c\rightarrow p}: x_c\rightarrow x_p=\left[\sum_{k=1}^{K+1}W^k_c(x_c)B_{c\rightarrow p}\right]x_c$    (1)

### Inverse forward skinning
Inversing eq.(1) leads to

$\mathcal{S}_{p\rightarrow c}: x_p\rightarrow x_c=\left[\sum_{k=1}^{K}W^k_c(x_c)B_{c\rightarrow p}\right]^{-1}x_p$  (2)

A chick-and-egg problem.

### Backward skinning 

$\mathcal{S}_{p\rightarrow c}: x_p\rightarrow x_c=\left[\sum_{k=1}^{K}W^k_p(x_p|\theta)B_{p\rightarrow c}\right]x_p = \left[\sum_{k=1}^{K}W^k_p(x_p|\theta)B_{c\rightarrow p}^{-1}\right]x_p$   (3)

Here we need to define another set of weights for each pose, $W_p(x_p|
\theta)$ 
**Note that inverse matrix is not distributive.** (2) and (3) are trivially different.

## Methods 
We denote the approximation of $\mathcal{S}_{p\rightarrow c}$ by $\mathcal{S}'_{p\rightarrow c}$  
### TAVA(UCB,2022) / SNARF(ETH,2021): Iterative root findings (Forward skinning)
For each $x_p$, find $\arg\min_{x_c} ||\mathcal{S}_{c\rightarrow p}(x_c)-x_p||^2$. TAVA/SNARF considers non-rigid delta added on the **deformed point**, which we denote as $\Delta_{\phi_{\Delta}}(x_c)$. The forward weight $W_c$ is learnable via a neural net $\phi_c$. So TAVA/SNARF searches for
$$\mathcal{S}'_{p\rightarrow c}(x_p) = \arg\min_{x_c}||\sum_{k=1}^{K+1}W^k_{\phi_c}(x_c)x_c+\Delta_{\phi_{\Delta}}(x_c)-x_p||^2$$
* Handling background: $B_{c\rightarrow p}^{K+1}=I$

### HumanNeRF(Google,2021): Neural weight approximation (Backward skinning)
$W_p(x_p|B_{p\rightarrow c}) = \textrm{softmax}(\{W_{\phi_c}^k(B^k_{p\rightarrow c} x_p),k=1,\ldots,K\})$

$\mathcal{S}'_{p\rightarrow c}(x_p)=\left[\sum_{k=1}^{K}W^k_p(x_p|\theta)B_{p\rightarrow c}\right]x_p$
* Handling background: during volumne rendering: weight alpha via the foreground probability, i.e. $(1-\sum_{k=1}^KW_{\phi_c}^k(B^k_{p\rightarrow c} x_p)) $

### NiLBS(UT,2020): Neural weight approximation (Inverse forward skinning)
NiLBS uses a learnable neural function $W_{\phi_p}(x_p|\theta)$ to **approximate** $W_c(x_c)$. 
To simplify learning, NiLBS projects each posed query coordinate to the local coordinate frame of each bone - The input $(x_p|\theta)$ to $W_{\phi_p}$ is first mapped to $\{\left[B_p^k\right]^{-1}x_p,k=1,\ldots,K\}$. 
$$\mathcal{S}'_{p\rightarrow c}(x_p) = \left[\sum_{k=1}^{K}W^k_{\phi_p}(x_p|\theta)\right]^{-1}x_p$$

Training $W_{\phi_p}$ suffers from data sparsity.
* Handling background: $B_{c\rightarrow p}^{K+1}=I$

### NeuralBody-Minutes (ZJU,2023) - Use SMPL weights (Inverse forward skinning)
For $x_p$, search the nearest point on the registered SMPL mesh as an approximation of $S_{p\rightarrow c}(x_p)$ and obtain its skinning weight $W_{p\rightarrow c}'(x_p)$ via interpolation. Then use **inverse linear skinning**
$$S'_{p\rightarrow c}(x_p)=\left[\sum_{k=1}^{K}W'_{p\rightarrow c}(x_p)B_{c\rightarrow p}\right]^{-1}(x_p)$$
* Handling Background ?




