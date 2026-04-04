# 前馈高斯插点网络 Pipeline 详解

## Network Role in the Full System

前馈高斯插点网络在整个系统中扮演的角色是：**当增量式 2DGS mapping 系统接收到新关键帧时，对经过上游筛选（SPNet 补点 + 渲染透明度/颜色误差过滤 + 颜色梯度过滤 + 体素下采样）后确定需要新增的候选 3D 点，直接预测其完整的 2DGS surfel 属性。**

网络的输入输出边界清晰：
- **输入**：一组候选 seed 点（含 3D 位置、2D 像素坐标、逆深度、法线、颜色、间距等几何先验）+ 当前帧及历史帧的原始/渲染图像。
- **输出**：每个 seed 点的 2DGS 属性——2D 尺度、旋转四元数、不透明度、SH 颜色、世界坐标位置。
- 网络**不负责**决定"在哪里插点"（这是上游筛选模块的职责），也**不负责**后续的在线优化迭代。

在代码中，网络推理发生在 `gaussian.cpp` 的 `initialize()` 和 `extend()` 函数内：`prepare_rg_input()` 构造输入 → `regress_gaussians()` 执行推理并后处理 → `densificationPostfix()` 将预测结果写入高斯地图。

---

## Inputs and Context Construction

### 候选 Seed 的几何先验

每个候选 seed 点携带以下从上游模块获得的先验信息：

| 先验 | 维度 | 来源 | 用途 |
|------|------|------|------|
| `base_pixel` | (N, 2) | LiDAR 投影 / SPNet 采样 | 点在当前帧的 2D 像素坐标 |
| `inv_depth` | (N, 1) | LiDAR 深度 / SPNet 补全深度 | 逆深度，用于尺度归一化和位置参数化 |
| `n_cam` | (N, 3) | DA3 法线引导（`build_da3_guidance()`） | 相机空间法线，用于旋转构造的锚点 |
| `mask` | (N, 1) | DA3 有效掩码 | 法线/深度是否有效 |
| `dis` | (N, 1) | KNN 像素距离（`compute_knn_distance()`） | 局部点密度先验，用于尺度预测 |
| `base_sh` | (N, 3) | 单帧 RGB 观测 → `RGB2SH()` | DC 颜色先验 |
| `R_wc` | (3, 3) | 当前相机的 camera-to-world 旋转 | 旋转从相机系转世界系 |

### 特征提取：Shared Frozen Backbone

系统使用一个**共享权重、冻结参数**的 CNN backbone，对所有图像统一提取逐像素特征图。

- **架构**：ResNet（多尺度特征投影 + 求和，InstanceNorm，输出 128 维）或 DINO 混合体（ResNet + ViT local/global token 融合）。代码在 `src/model/backbone/backbone_resnet.py` 和 `backbone_dino.py`。
- **权重来源**：从 PixelSplat 预训练检查点中提取（`scripts/extract_backbone.py`），通过 TorchScript trace 导出为 `.pt` 文件。
- **推理封装**：`src/model/inference_wrapper.h/cpp` 中的 `FeatureExtractor` 类，接受 `(1, 3, H, W)` 图像，输出 `(1, 128, H, W)` 特征图。

Backbone 作用于以下 4 类图像：
1. **当前帧原始图像** `I_t` → `F_t^{img}`
2. **当前帧渲染图像** `R_t`（用当前高斯地图从当前视角渲染） → `F_t^{ren}`
3. **历史帧原始图像** `{I_{t-k}}` → `{F_{t-k}^{img}}`
4. **历史帧渲染图像** `{R_{t-k}}`（用**当前**高斯地图从**历史**视角渲染） → `{F_{t-k}^{ren}}`

### Point-Aligned Context Construction

这是网络输入准备的核心步骤，实现在 `feature_utils.h::compute_context_features()` 中（约 175 行）。其关键特征是：**所有特征采样都是以候选 seed 点为中心的逐点操作（point-aligned），而非全图全局编码。**

#### 当前帧特征采样

1. 将候选点的 `base_pixel` 坐标归一化到 `[-1, 1]` 范围。
2. 使用 `torch::nn::functional::grid_sample`（双线性插值，`align_corners=true`，零填充）从 `F_t^{img}` 采样 → `f_curr` (N, 128)。
3. 从 `F_t^{ren}` 在同一像素位置采样 → `render_f_curr` (N, 128)。
4. 计算 `curr_dir = normalize(R_cw · P_world + t_cw)` → 当前相机空间下的归一化观察方向 (N, 3)。

#### 历史帧投影、可见性判断与特征采样

对滑窗内 W 帧历史关键帧（向当前帧之前回溯），逐帧执行：

1. **投影**：将候选点的世界坐标通过历史帧的 `R_cw_hist × P_world + t_cw_hist` 投影到历史帧图像平面，得到历史帧下的像素坐标。
2. **可见性判断**：检查 `z > 0.1`（在相机前方）且像素在 `[0, W) × [0, H)` 内 → `hist_mask[:, w, :] ∈ {0, 1}`，(N, W, 1)。
3. **原始特征采样**：从 `F_{t-k}^{img}` 按投影坐标 `grid_sample` → `hist_f[:, w, :] ∈ R^{N×128}`，不可见点置零。
4. **渲染特征采样**：用**当前**高斯模型从该**历史**视角渲染一张图，过 backbone 提取特征，`grid_sample` → `hist_render_f[:, w, :] ∈ R^{N×128}`。这捕捉了"当前模型对历史视角的预测"。
5. **方向计算**：从历史相机中心指向候选点的方向，**统一到当前相机坐标系**表达：`hist_dir[:, w, :] = normalize(R_cw_current · (P_world - C_hist))` (N, 3)。统一坐标系使 cross-attention 中不同帧的方向具有可比性。

#### Current vs History vs Rendered Context 的角色

| Context 来源 | 承载的信息 | 角色 |
|-------------|-----------|------|
| `f_curr` (当前帧原始) | 当前帧在该点的真实场景外观 | 提供"场景长什么样"的信号 |
| `render_f_curr` (当前帧渲染) | 当前地图在该点渲染出的外观 | 提供"当前地图认为这里长什么样"——与 f_curr 的差异指示重建不足程度 |
| `hist_f` (历史帧原始) | 同一点在历史帧中的真实外观 | 提供多视角外观变化信息（视角依赖效应、遮挡变化） |
| `hist_render_f` (历史帧渲染) | 当前地图从历史视角渲染的外观 | 提供"地图在其他视角的表现"，帮助网络判断地图的全局一致性 |
| `curr_dir / hist_dir` (观察方向) | 各帧到该点的观察方向 | 编码视差和视角依赖性，参与 cross-attention 权重计算 |
| `hist_mask` (可见性) | 该点在各历史帧是否可见 | attention mask，确保不可见帧不参与聚合 |

#### Relative Poses 如何进入网络语义

系统不直接将位姿矩阵作为网络输入。相对位姿信息通过以下方式隐式进入：

1. **投影关系**：历史帧特征采样时，投影坐标本身编码了当前→历史的相对位姿。
2. **观察方向**：`curr_dir` 和 `hist_dir` 本质上是相对位姿的几何表达——同一点在不同帧下的归一化方向向量包含了视差信息。
3. **R_wc 直接参与旋转解码**：`R_wc` 不进入 MLP 编码器，但在最终旋转构造时用于将相机系旋转转到世界系（`R_world = R_wc × R_cam`）。

---

## Core Network Architecture

### Cross-Attention Context Fusion

实现在 `gaussian_mlp.py::CrossAttentionContext` 中，是一个**单头 cross-attention** 模块，将 W 帧历史上下文聚合为固定维度的上下文向量。

**Query 构造**：
```
q_input = [curr_dir(3) ‖ f_curr(128) ‖ render_f_curr(128)] → W_q → q ∈ R^{N×1×64}
```

**Key 构造**：
```
k_input = [hist_dir(3) ‖ hist_f(128) ‖ hist_render_f(128)]  (per slot)
→ W_k → k ∈ R^{N×W×64}
```

**Value 构造**：
```
v_input = [hist_dir(3) ‖ hist_f(128) ‖ hist_render_f(128)]  (per slot)
→ W_v → v ∈ R^{N×W×256}
```

**Attention 计算**：
```
scores = (q · k^T) / √64            → (N, 1, W)
scores[hist_mask == 0] = -1e9       → 不可见帧被屏蔽
weights = softmax(scores, dim=-1)    → (N, 1, W)
ctx_agg = (weights · v).squeeze(1)   → (N, 256)
```

**关键设计点**：观察方向（`curr_dir`, `hist_dir`）显式参与 Q/K 的投影，使注意力权重能感知视差关系——靠近当前视角的历史帧自然获得更高权重，严重遮挡的帧被 mask 掉。这比简单的特征拼接或平均池化更能捕捉 view-dependent 的上下文。

### Geometry-Aware Encoder

Cross-attention 输出后，与其他几何先验拼接并输入残差 MLP。

**位置编码分支**（`gaussian_mlp.py::get_embedder()`）：

| 输入 | 变换 | 频带数 | 输出维度 |
|------|------|--------|---------|
| `inv_depth` (1) | `γ(v) = [v, sin(2^i π v), cos(2^i π v)]` | 4 | 9 |
| `n_cam` (3) | 同上 | 2 | 15 |
| `dis` (1) | 先 `log(·)` 再 PE | 3 | 7 |

频带为 `2^{0}, 2^{1}, ...`，预计算并存储为 registered buffer（TorchScript 兼容）。

**拼接构造**（总 544 维）：
```
input = [f_curr(128) ‖ render_f_curr(128) ‖ ctx_agg(256) ‖
         PE(inv_depth)(9) ‖ PE(n_cam)(15) ‖ mask(1) ‖ PE(log(dis))(7)]
```

**残差 MLP 编码器**：
```
Linear(544 → 512) → ReLU → [ResidualBlock(512)]×4
```
每个 `ResidualBlock`：`x + ReLU(Linear(512 → 512))`——单层 pre-activation 残差连接。输出 512 维特征向量 `feat`。

---

## Prediction Heads and Parameterization

编码器输出的 512 维 `feat` 分支到 9 个独立预测头。每个 `PredictionHead` 的通用架构为 `Linear(512→d_hidden) → ReLU → Linear(d_hidden→d_out)`（默认 `d_hidden=512`，部分头使用 256）。

### Head 1: 2D Scale Head
- **输出**：(N, 2)——2D surfel 切平面两个方向的尺度。
- **激活**：`exp(raw) × (dis / 2)`。`dis` 是 KNN 像素距离，提供局部密度自适应。
- **后处理**（C++ 端）：`scale_world = scale_pixel / (inv_depth × focal × scale_modifier)`，从图像平面尺度转到世界尺度。

### Head 2-4: Rotation Heads（Normal-Guided Rotation Construction）

这是最精密的预测分支，由三个头协同构造旋转：

- **Delta z Head** → `δz ∈ R^{N×3}`：法线修正向量。
- **Auxiliary a Head** → `aux_a ∈ R^{N×3}`：辅助向量（bias 初始化为 0.1 避免退化）。
- **Lambda Head** → `λ = softplus(raw) ∈ R^{N×1}`：修正强度（bias 初始化为 −2.0，使 `softplus(-2) ≈ 0.13`，初始紧靠法线先验）。

**构造过程**：
1. `z_unnorm = n_cam + λ · δz` → 修正后的法线方向
2. `z_new = normalize(z_unnorm)` → surfel 法线（z 轴）
3. `y_cam = normalize(cross(z_new, aux_a))` → surfel y 轴
4. `x_cam = cross(y_cam, z_new)` → surfel x 轴
5. `R_cam = [x_cam | y_cam | z_new]` → 相机系旋转矩阵
6. `R_world = R_wc × R_cam` → 世界系旋转矩阵
7. Shepperd 方法 → 四元数 `(w, x, y, z)`（4-case 分支，全向量化，GPU 友好）

### Head 5: Opacity Head
- **输出**：`sigmoid(raw) ∈ (0, 1)`——真实不透明度。
- **后处理**（C++ 端）：`inverse_sigmoid(α) × opacity_modifier` → logit 空间，供 2DGS 渲染器使用。

### Head 6: Color DC Head
- **输出**：(N, 1, 3)——DC SH 系数**残差**。
- **计算**：`pred_dc = head(feat) + base_sh`——加到从原始 RGB 观测推算的 `base_sh` 先验上。零初始化确保从先验出发。

### Head 7: Color Rest Head
- **输出**：(N, 15, 3) = 45 维——1–3 阶 SH 系数，直接预测。

### Head 8: Delta Pixel Head
- **输出**：(N, 2)——亚像素位置修正。
- **初始化**：权重和 bias 均为零（`nn.init.zeros_`），确保初始位置完全保留 `base_pixel` 先验。

### Head 9: Delta Inv-Depth Head
- **输出**：(N, 1)——逆深度修正。
- **初始化**：同上，零初始化。

---

## How Outputs Are Converted into Inserted 2DGS Gaussians

网络输出经过 `gaussian.cpp::regress_gaussians()` 的后处理，转换为最终的 2DGS surfel 属性：

### 位置重构（2.5D → 3D World）
```
u_new = base_pixel_x + delta_pixel_x
v_new = base_pixel_y + delta_pixel_y
z_cam = 1 / (inv_depth + delta_inv_depth)
x_cam = (u_new - cx) × z_cam / fx
y_cam = (v_new - cy) × z_cam / fy
P_world = R_wc × [x_cam, y_cam, z_cam]^T + t_wc
```

### 尺度转换
```
scale_world = scale_pixel / (inv_depth × focal × scale_modifier)
scaling_ = log(scale_world)   // 存储为 log 空间
```

### 旋转
直接使用 normal-guided construction 的输出四元数（已在世界系），存入 `rotation_`。

### 不透明度
```
opacity_ = inverse_sigmoid(sigmoid(raw) × opacity_modifier)
```
二次变换确保最终值在 logit 空间且受 `opacity_modifier` 调节。

### 颜色
```
features_dc_ = pred_dc   // base_sh + residual
features_rest_ = pred_rest
```

所有属性 `cat` 到现有高斯张量后面（`densificationPostfix()`），设 `requires_grad_(true)`，分配全局唯一 ID，清空原始点云缓存。新高斯立即参与后续的在线优化迭代。

---

## Why This Is Not Full Feed-Forward Reconstruction

需要明确区分本网络与 pixelSplat、MVSplat 等"全前馈 3D 重建"方法的本质差异：

| 维度 | 全前馈重建 (pixelSplat 等) | 本网络 |
|------|--------------------------|--------|
| 目标 | 从少量图像直接重建完整场景 | 为增量式 SLAM 的逐帧补点提供高质量初始化 |
| 输入 | 一组稀疏视图图像 | 候选 seed 点 + 多帧上下文特征 + 几何先验 |
| 输出 | 整个场景的高斯集合 | 仅新增高斯的属性（增量式） |
| 是否还需优化 | 通常不需要 | 是——网络输出是初始值，后续参与在线光度优化 |
| 训练数据 | 外部大规模多视图数据集 | 自蒸馏——系统自身在线优化的中间结果 |
| 操作粒度 | 图像级（每像素一个或多个高斯） | 点级（仅对候选 seed 点预测） |
| 上游依赖 | 无 / 仅依赖位姿 | 依赖 SPNet 补点、DA3 法线、渲染过滤等上游模块 |

本网络的定位是"learned initialization"，而非"learned reconstruction"。它加速了收敛但不取消迭代优化；它利用了 SLAM 系统的时序上下文但不独立工作。

---

## 适合画网络图的模块划分建议

以下是按照从左到右数据流方向的模块划分方案，适合直接用于画网络结构图。

### Band 1: Multi-Source Image Inputs + Shared Backbone（顶部/左侧）

**模块 A: 输入**
- Candidate Seeds（小块）：附带 base_pixel, inv_depth, n_cam, mask, dis, R_wc, base_sh
- 4 类图像输入：`I_t`, `R_t`, `{I_{t-k}}`, `{R_{t-k}}`

**模块 B: Shared Frozen Backbone**
- 一个统一的 CNN 方块
- 4 条输出箭头 → 4 组特征图：`F_t^{img}`, `F_t^{ren}`, `{F_{t-k}^{img}}`, `{F_{t-k}^{ren}}`

### Band 2: Point-Aligned Context Construction（中左）

**模块 C: Current Sampling**
- `base_pixel` → `grid_sample(F_t^{img})` → `f_curr` (N×128)
- `base_pixel` → `grid_sample(F_t^{ren})` → `render_f_curr` (N×128)
- 相机变换 → `curr_dir` (N×3)

**模块 D: History Projection & Visibility**
- World point → 历史相机投影 → 像素坐标
- 可见性判断 → `hist_mask` (N×W×1)

**模块 E: History Sampling**
- 投影坐标 → `grid_sample({F_{t-k}^{img}})` → `hist_f` (N×W×128)
- 投影坐标 → `grid_sample({F_{t-k}^{ren}})` → `hist_render_f` (N×W×128)
- 方向计算 → `hist_dir` (N×W×3)

### Band 3: Cross-Attention Context Fusion（中央核心——最大最显眼）

**模块 F: Cross-Attention Context**
- Q branch: `[curr_dir ‖ f_curr ‖ render_f_curr]` → `W_q` → Q (N×1×64)
- K branch: `[hist_dir ‖ hist_f ‖ hist_render_f]` → `W_k` → K (N×W×64)
- V branch: 同 K 输入 → `W_v` → V (N×W×256)
- Attention Mask: `hist_mask` → `-∞` 屏蔽
- `softmax(QK^T/√64) · V` → `ctx_agg` (N×256)

### Band 4: Geometry-Aware Encoder（中右）

**模块 G: Positional Encoding**
- `PE(inv_depth)` → 9D
- `PE(n_cam)` → 15D
- `log + PE(dis)` → 7D

**模块 H: Concat + Residual MLP Encoder**
- 拼接：`f_curr(128) + render_f_curr(128) + ctx_agg(256) + PE×3(31) + mask(1)` = **544D**
- `Linear(544→512) → ReLU → ResidualBlock×4`
- 输出：`feat` (N×512)

### Band 5: Multi-Head Prediction（底部左）

**模块 I: 9 个预测头**
- 扇出结构，从 `feat` 分支：
  - 2D Scale Head → (N, 2)
  - Delta z Head → (N, 3)
  - Aux a Head → (N, 3)
  - Lambda Head → (N, 1)
  - Opacity Head → (N, 1)
  - Color DC Head → (N, 3)
  - Color Rest Head → (N, 45)
  - Delta Pixel Head → (N, 2)
  - Delta Inv-Depth Head → (N, 1)

### Band 6: Constrained Geometric Decoding（底部右）

**模块 J: Scale Decoding**
- `exp(raw) × dis/2` → pixel scale → `÷(inv_depth × focal)` → world scale

**模块 K: Normal-Guided Rotation**
- `n_cam + λ·δz` → `z_new` → `cross(z_new, aux_a)` → `y_cam` → `R_cam` → `R_wc × R_cam` → quaternion

**模块 L: Color Assembly**
- `base_sh + DC_residual` → final DC
- SH rest 直接输出

**模块 M: 2.5D Position Decoding**
- `base_pixel + Δpixel` + `inv_depth + Δinv_depth` → 反投影 → `R_wc × P_cam + t_wc` → world position

### 最终输出

**模块 N: Predicted 2D Surfel Gaussians**
- 薄片状 surfel 图示（不是球体）
- 属性标注：`s` (2D scale), `r` (rotation), `α` (opacity), `c` (SH color), `p` (position)
- → **Incremental 2DGS Map**（终点小块）

### 配色建议

| 模块 | 推荐色系 |
|------|---------|
| 输入图像 + Candidate Seeds | 浅蓝/灰蓝 |
| Shared Backbone + Feature Maps | 浅青蓝 |
| Point-Aligned Context Construction | 浅青绿 |
| Cross-Attention Core | **暖橙/杏色（重点高亮）** |
| Residual MLP Encoder | 浅粉橙 |
| Multi-Head Prediction | 浅粉/浅黄/浅橙分块 |
| Constrained Geometric Decoding | 浅紫蓝 |
| Final 2D Surfels | 浅蓝紫 |
