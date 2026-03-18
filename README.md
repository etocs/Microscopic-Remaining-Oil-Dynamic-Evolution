# Microscopic Remaining Oil Dynamic Evolution

基于动静双模态融合与达西定律物理约束的混合神经网络，实现微观剩余油时间序列动态演变预测。

## 模型设计
- **静态端（CNN3D）**：提取孔隙结构与剩余油空间分布特征（自适应池化支持 256×256×64 等尺寸）。
- **动态端（BiLSTM）**：捕捉玻璃刻蚀驱替时序（统一 30 个时间步，注入压力 / 含水率 / PV 等）。
- **特征融合 + 通道注意力**：拼接 1024 维融合特征并自适应加权关键通道。
- **达西定律物理修正**：v = μ K ∇P，速度输出软约束；复合损失 = 预测误差 + λ·Darcy 一致性。
- **输出**：剩余油饱和度（∈[0,1]）、渗流速度（≥0）、动静态转换概率。

## 预处理
- `static_image_preprocess(volume, target_shape=(256,256,64))`
  - 灰度标准化 → 异常裁剪 → 三线性插值到目标体素，输出张量形状 `(1, D, H, W)`.
- `dynamic_time_series_preprocess(sequence, target_steps=30)`
  - 线性插值补缺失 → 对齐 30 步 → 每特征 Min-Max 归一化。
- `pair_modalities(static_features, dynamic_features, similarity_matrix=None)`
  - 按孔隙结构相似性（可选相似度矩阵）配对动静态样本。

## 快速开始
```python
import torch
from microscopic_remaining_oil_dynamic_evolution.model import FusionModel, PhysicalConstraintLoss

model = FusionModel(static_channels=1, dynamic_features=5, physics_weight=0.05)
criterion = PhysicalConstraintLoss(lambda_darcy=0.05)

static_volume = torch.randn(4, 1, 64, 128, 128)      # 3D 图像
dynamic_seq = torch.randn(4, 30, 5)                  # 时序特征
permeability = torch.full((4, 1), 0.2)
viscosity = torch.full((4, 1), 1.0)
pressure_grad = torch.full((4, 1), 0.5)
targets = {
    "saturation": torch.rand(4, 1),
    "velocity": torch.rand(4, 1),
    "transition_prob": torch.rand(4, 1),
}

outputs = model(static_volume, dynamic_seq, permeability, viscosity, pressure_grad)
loss = criterion(outputs, targets, permeability, viscosity, pressure_grad)
loss.backward()
```

## 测试
```bash
python -m unittest discover
```
