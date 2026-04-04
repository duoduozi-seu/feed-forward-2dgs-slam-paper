# 全量工作总结：从 Baseline 到当前系统的系统级演化

## Overview

本项目以 Gaussian-LIC / Gaussian-LIC2 开源系统为基础（baseline commit `a77558f`），在 `feed-forward-gaussian` 分支上经历约 105 次提交，完成了从"基于 3DGS 的增量式 LiDAR-Inertial-Camera 高斯地图重建"到"融合 2DGS 主干、前馈式高斯初始化网络、稠密几何先验与位姿反向优化的全新增量式重建系统"的系统级转变。以下从系统层面进行全面、基于代码证据的总结。

---

## Baseline vs Current：系统级本质变化

### Baseline 系统本质

Baseline 是 Gaussian-LIC 的一个工作版本：

- **表示**：标准 3D Gaussian Splatting（3DGS），每个高斯具有 3 维尺度（椭球体）。
- **初始化**：完全依赖 LiDAR 投影点，手工设定尺度（基于深度）、不透明度（固定低值）、颜色（单帧观测 DC）。无深度补全。
- **致密化**：标准 ADC（adaptive density control），基于梯度范数随机扰动进行 clone/split。
- **优化**：简单的关键帧滑窗 + 固定视图选择策略。
- **位姿**：完全依赖外部 LIO 前端，无反向位姿优化。
- **渲染器**：原版 3DGS tile-based rasterizer。
- **前馈网络**：不存在。

`gaussian.cpp` 约 1060 行，`mapping.cpp` 约 242 行。系统功能单一，本质上是"将 LiDAR 点变成 3DGS 椭球 + 在线光度优化"。

### Current 系统本质

当前系统是一个多模态融合的增量式 2DGS 场景重建系统：

- **表示**：2D Gaussian Splatting（2DGS），每个高斯为 surfel（2 维尺度的扁平面片），具有几何意义的法线方向。
- **初始化**：前馈式神经网络直接预测所有高斯属性（尺度、旋转、不透明度、SH 颜色、位置），或传统模式下融合 SPNet 稠密深度 + DA3 法线引导。
- **致密化**：方向性 2D 梯度驱动的 clone/split + 共视感知调度 + 新生高斯梯度增强。
- **优化**：多优先级训练视图选择（滑窗 + 致密化帧 + 高损失帧 + 随机帧）。
- **位姿**：基于 2DGS 渲染残差的 IEKF 位姿反向优化，带完备稳定性保护。
- **渲染器**：完整移植并适配的 2DGS surfel rasterizer（`diff_surfel_rasterization_2d/`）。
- **前馈网络**：完整的 backbone + cross-attention + residual MLP + 多头预测 + 几何约束解码的前馈管线，以及配套的自蒸馏数据集生成 + 离线训练 + 部署闭环。

`gaussian.cpp` 增长至约 3500 行，`mapping.cpp` 增至约 660 行。新增整个 `src/model/` 子系统（~2400 行 C++ / ~1400 行 Python）。系统从一个简单的在线高斯优化工具，演化为一个"几何先验驱动 + 前馈网络预测 + 在线迭代优化"三位一体的增量式重建系统。

---

## 主要完成工作

### 一、2DGS 主干替换与全链路重构

**工作内容**：将 3DGS 椭球体表示替换为 2DGS surfel（2D Gaussian Splatting），涉及渲染器、模型存储、初始化、优化、致密化的全链路重写。

**关键技术变化**：

- **尺度维度**：`scaling_` 从 `(N, 3)` 变为 `(N, 2)`，只有切平面内两个方向的尺度（`gaussian.h:136`）。
- **渲染器**：完整移植 `diff_surfel_rasterization_2d/` 模块，包含 CUDA 前向/反向传播内核、tile-based surfel rasterizer。新增 `rasterizer_2d.cpp/h`、`renderer_2d.cpp/h` 共约 1100 行。
- **渲染输出扩展**：2DGS 渲染器额外输出法线图（`rendered_normal`）、中值深度图（`median_depth`）、失真度图（`distortion`），用于几何约束损失。
- **特征合并**：`getFeatures()` 将 DC 与 rest SH 合并为 `(N, K, 3)` 张量，适配 2DGS 渲染器接口（`gaussian.cpp:599`）。
- **损失函数**：在原有光度损失（L1 + SSIM）基础上新增 `lambda_dist × distortion_loss + lambda_normal × normal_consistency_loss`（`gaussian.cpp` optimize 函数内），确保 surfel 几何一致性。
- **旋转初始化**：引入 `compute_rotation_from_normals()`（`feature_utils.h`），将 DA3 法线转换为 surfel 朝向四元数，使初始 surfel 法线与表面法线对齐。

**系统意义**：2DGS 的 surfel 表示赋予高斯几何意义——每个 surfel 有明确法线方向，可参与法线一致性损失和失真约束，为几何先验引入（DA3 法线引导）和前馈网络的 geometry-aware rotation construction 提供了必要基础。这不是简单的渲染器替换，而是整个表示范式的升级。

### 二、前馈式高斯初始化/插点网络

**工作内容**：设计并实现了一个完整的前馈网络，用于在增量式重建中对新关键帧的候选点直接预测 2DGS 属性，替代传统手工初始化。

**核心模块**：

- **特征提取器**（`src/model/backbone/`）：从 PixelSplat 预训练权重中提取的 ResNet/DINO backbone，通过 TorchScript 导出后在 C++ 中推理。支持多尺度特征投影 + 求和融合，输出 128 维逐像素特征图。
- **逐点上下文构造**（`feature_utils.h::compute_context_features()`，约 175 行）：对每个候选 3D 点，从当前帧原始图与渲染图的特征图上按像素位置采样（`grid_sample`），并从滑窗内 W 帧历史帧的特征图上逐帧投影、可见性判断、采样，构建多视角上下文。同时计算各帧下的观察方向（统一到当前相机坐标系）。
- **跨注意力上下文融合**（`gaussian_mlp.py::CrossAttentionContext`）：Q 来自当前帧方向 + 特征，K/V 来自历史帧方向 + 特征，带可见性掩码的 softmax attention，输出 256 维聚合上下文。
- **几何感知残差 MLP 编码器**（`gaussian_mlp.py`）：将 `f_curr(128) + render_f_curr(128) + ctx_agg(256) + PE(inv_depth)(9) + PE(n_cam)(15) + mask(1) + PE(dis)(7)` 共 544 维拼接输入 → `Linear(544→512) + ReLU + ResidualBlock×4` → 512 维特征。
- **多头预测**（`gaussian_mlp.py`）：9 个独立预测头，分别输出 2D 尺度、旋转残差（δz, aux_a, λ）、不透明度、DC 颜色残差、高阶 SH、像素偏移、逆深度偏移。
- **几何约束解码**（`gaussian_mlp.py + gaussian.cpp::regress_gaussians()`）：
  - 尺度：`exp(raw) × (dis/2)`，再除以 `inv_depth × focal` 转世界尺度。
  - 旋转：法线引导构造——`z_new = normalize(n_cam + λ·δz)`，cross product 构造正交基，`R_world = R_wc × R_cam`，Shepperd 方法转四元数。
  - 位置：2.5D 参数化——`(base_pixel + Δpixel, inv_depth + Δinv_depth)` 反投影到世界坐标。
  - 颜色：`base_SH + DC_residual`，保留 SH rest 直接预测。
- **C++ 推理封装**（`inference_wrapper.cpp/h`）：通过 LibTorch 加载 TorchScript 模型，`GaussianRegressor::regress()` 接口。

**系统意义**：这是系统中最核心的创新模块。它将候选高斯的初始化从"手工规则"升级为"学习预测"，利用多视角上下文和几何先验，在不依赖迭代优化的情况下直接给出高质量的 2DGS 属性初始值，显著加速收敛并提升初始质量。

### 三、稠密几何先验融合（SPNet + DA3）

**工作内容**：引入两个独立的稠密深度/法线先验网络，服务于不同目的。

- **SPNet 深度补全**（`depth_completion.cpp/h`，约 913 行；`spnet_wrapper.cpp/h`）：实现 Gaussian-LIC2 论文中的 Algorithm 1——SPNet 稀疏深度补全 → 失败检测（ε₁ 阈值）→ 深度梯度边缘过滤 → LiDAR mask 膨胀 → 30×30 patch 采样 → 反投影生成补充 3D 点。为 LiDAR 盲区提供稠密 3D 补充点。
- **DA3 法线引导**（`da3_wrapper.cpp/h`；`depth_completion.cpp::build_da3_guidance()`）：Depth-Anything-v2 推理 → 仿射对齐（`a×x+b` 优先，回退到纯尺度 `a×x`）→ 有效性过滤（置信度 + 深度范围）→ 中心差分计算法线图 → 法线翻转确保朝向相机 → 按像素采样。为 surfel 提供初始法线方向，驱动旋转初始化。
- **两者分工明确**：SPNet 负责"在哪里补点"（空间分布），DA3 负责"surfel 朝哪个方向"（几何朝向）。SPNet 补点不使用 DA3 深度，DA3 法线不影响 SPNet 采样策略。

**系统意义**：SPNet 解决了稀疏 LiDAR 覆盖不足的问题，DA3 为 2DGS surfel 提供了有意义的几何先验——不需要等待优化收敛就能给出合理的 surfel 朝向，与前馈网络的 normal-guided rotation construction 形成呼应。

### 四、自蒸馏数据集生成与离线训练闭环

**工作内容**：构建了从在线 SLAM 运行中自动采集训练数据、到离线训练前馈网络、到部署回在线系统的完整闭环。

- **在线采集**（`gaussian.cpp::captureDatasetTargetSnapshot()` + `saveDataset()`）：
  - 在线运行时，当某帧的高斯训练到 `dataset_target_train_times` 次时快照中间状态作为伪 GT。
  - 终止时回溯所有关键帧，重新运行 backbone + context 特征构造（确保因果正确性——只用该帧加入前已有的高斯做渲染特征）。
  - 保存帧级 `.pt`（包含 15 个张量：当前/历史的原始/渲染特征、方向、mask、逆深度、法线、ID 等）。
- **数据集准备**（`scripts/prepare_shards.py`，~763 行）：
  - 加载收敛后的 PLY（伪 GT），匹配 frame `.pt` 中的 `added_ids` → GT 属性。
  - 关键预处理：强制 `scale_x ≥ scale_y`，消除尺度-旋转歧义（交换尺度时同步旋转四元数 90°）。
  - 过滤 NaN/Inf，场景级随机打乱，train/val 分割，分桶保存。
- **流式数据加载**（`scripts/dataset_sharded.py`）：基于 shuffle buffer 的 `IterableDataset`，支持多 worker、跨 shard 打乱。
- **离线训练**（`scripts/train_regressor.py`，~517 行）：
  - 6 种 loss：scale（log 世界尺度 L1）、rotation（四元数余弦距离）、color DC（MSE）、color rest（MSE）、opacity（L1）、position（2.5D 像素 + 逆深度 L1）。
  - Adam 优化，梯度裁剪，NaN 检测与跳过。
  - Backbone 提取：`scripts/extract_backbone.py` 从 PixelSplat checkpoint 中提取并 trace。
- **模型导出**（`scripts/export_models.py`）：TorchScript trace 导出，供 C++ LibTorch 加载。

**系统意义**：这是一套完整的"系统自我进化"管线——在线系统运行产生训练数据，训练出前馈网络后反过来增强在线系统。这种自蒸馏范式消除了对外部标注数据集的依赖，使系统能在任意新场景上迭代提升。

### 五、基于 2DGS 渲染的 IEKF 位姿反向优化

**工作内容**：实现了一个完整的迭代扩展卡尔曼滤波（IEKF）位姿优化器，利用 2DGS 渲染残差对外部前端给出的位姿进行精化。

- **核心算法**（`pose_optimizer.cpp`，342 行）：
  - 输入：外部 LIO 前端的位姿预测作为先验（含协方差）。
  - 残差：当前帧 RGB 与 2DGS 渲染图的逐像素差异。
  - 线性化：由 `renderer_2d` 输出 JTJ/JTr（6×6 海塞近似 + 6 维梯度）。
  - 更新公式：`A = P⁻¹ + JTJ/σ², b = P⁻¹·η + JTr/σ²`，LDLT 求解 `δξ`。
  - SE(3) 更新：`T_cw⁺ = Exp(δξ^∧) · T_cw`。
- **稳定性保护**（多重 guard-rail）：
  - 最小有效像素数检查。
  - Alpha 覆盖率检查（避免在无高斯覆盖区域做优化）。
  - RMSE 恶化检测（候选解不能比当前解差 1% 以上）。
  - 单步旋转/平移步长裁剪（默认 0.3°/3cm）。
  - 病态矩阵检测。
  - 协方差正则化。
- **集成方式**（`mapping.cpp`）：在每帧 extend 之前运行；非关键帧只做位姿优化不做扩展/训练；关键帧的精化位姿用于后续 extend + optimize。

**系统意义**：利用已建高斯地图对位姿做后验修正，形成"位姿→地图→位姿"的正反馈环路。虽然当前版本默认关闭（`enable_pose_refinement: false`），但完整框架已就绪。

### 六、工程调度、筛点、致密化与评估工具

**训练视图调度**：多优先级分层选择（`gaussian.cpp` optimize 函数内）：
- P1：最新滑窗关键帧（最高优先级）。
- P_boost_forced：本轮被选为致密化的帧。
- P_boost_candidates：近期致密化帧的共视邻域（后致密化训练增强）。
- P2：高损失帧（`keyframe_loss > hiloss_threshold`）。
- P3：剩余帧随机。

**方向性致密化**：
- 累积的是 2D 屏幕空间梯度**向量** `(dx, dy)`（不仅是范数），保存在 `xyz_gradient_accum_ (N, 2)` 中。
- `computeWorldDir()` 将 2D 梯度方向反投影到 3D 世界方向。
- Clone/Split 沿该方向位移（而非随机扰动），位移量 `α = densify_alpha × max_scale`。
- 共视感知调度：同一轮被致密化帧的最小索引间隔 `densify_index_gap`；邻域上次致密化后需新增最小训练次数 `densify_min_train_after_covis`。

**新生高斯梯度增强**：新插入的高斯在 `newborn_boost_steps` 步内，位置梯度乘以 `densify_newborn_pos_lr_scale`（默认 2.0），加速其初始调整。

**扩展阶段多层筛选**（`extend()` 函数）：
- 渲染透明度过滤：`rendered_alpha < dynamic_threshold` 的位置需要新高斯。
- 颜色误差过滤：`error_map > hiColorLoss_threshold` 的位置需要新高斯。
- 暗区域自适应：暗区域（`dark_color_threshold`）降低 alpha 阈值 0.7×。
- 颜色梯度过滤（Sobel）：过滤低纹理区域。
- 体素下采样（`voxel_size`）。

**稀疏优化器**（SparseGaussianAdam）：每次只更新当前视图可见的高斯子集，配合 `densificationPostfix()` 动态扩展优化器状态。

**评估工具**：
- `evaluateVisualQuality()`：全量 PSNR/SSIM/LPIPS 评估（训练 + 测试帧）。
- `runTrainVisualEvalIfNeeded()`：训练过程中周期性记录指定帧的渲染质量曲线。
- `scripts/compare_train_visual_eval.py`：对比两次实验的质量指标演化。
- `scripts/visualize_mlp_predictions.py`：可视化前馈网络预测 → PLY 导出。
- `scripts/color_grad_thresh_helper.py`：辅助选择颜色梯度阈值。

---

## 系统根本性变化总结

| 维度 | Baseline | Current |
|------|----------|---------|
| 高斯表示 | 3DGS 椭球体 (3D scale) | 2DGS surfel (2D scale + 法线) |
| 初始化策略 | 手工规则 (depth-based scale, 固定 opacity) | 前馈网络预测 或 几何先验引导的传统模式 |
| 致密化 | 标准 ADC (随机扰动) | 方向性梯度驱动 + 共视感知调度 + 新生增强 |
| 深度先验 | 无 | SPNet 深度补全 + DA3 法线引导 |
| 位姿处理 | 完全依赖外部前端 | IEKF 反向优化（可选） |
| 训练策略 | 简单滑窗 | 多优先级分层视图选择 |
| 网络组件 | 无 | backbone + cross-attention MLP + 自蒸馏训练闭环 |
| 代码规模 | ~1300 行核心 | ~6500 行核心 + ~2700 行 Python |

---

## 简要评估

本项目从一个基础的 Gaussian-LIC 工作版本出发，完成了一次"从工程实践到方法论创新"的系统级跃升。最核心的贡献在于：**将增量式高斯地图重建中"新高斯如何初始化"这一问题从手工规则升级为学习驱动的前馈预测**，并围绕这一核心构建了完整的几何先验融合、自蒸馏训练、几何约束解码和 2DGS 适配体系。2DGS 主干替换不是孤立的渲染器更新，而是为整个 geometry-aware 方法论提供了表示层面的必要基础。
