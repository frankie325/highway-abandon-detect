# Highway Abandon Detect

高速公路抛洒物检测实验项目。

终端日志统一使用 Loguru，使用方法见 [Loguru 完整使用指南](docs/loguru.md)。

日志配置位于 `src/configs/model.yaml` 的 `logger` 节点：终端默认 INFO、文件默认 DEBUG，文件按时间命名写入项目根目录的 `logs/`，支持颜色、轮转、保留期限和压缩。

## 数据集图像变换策略

当前使用 Albumentations 处理单张图片及边界框，由 `RAODCollate` 为批次选择缩放尺寸、完成补边并生成掩码。处理流程为：

```text
PIL 图片转 RGB → 等比例缩放 → 训练时随机增强 → 归一化 → CHW 张量 → 批次右下补边
```

### 训练与验证

| 处理项 | 训练阶段 | 验证阶段 |
|---|---|---|
| 缩放策略 | 开启多尺度时，每个 batch 从 `[224, 512, 768]` 中随机选择一个最长边尺寸 | 最长边固定为 `512` |
| 长宽比 | 保持原始比例，同批图片共享目标最长边 | 保持原始比例 |
| 水平翻转 | 每张图片以 `0.5` 的概率执行，边界框同步翻转 | 关闭 |
| 亮度与对比度 | 每张图片以 `0.3` 的概率执行，两个变化范围参数均为 `(-0.1, 0.1)` | 关闭 |
| 图像裁剪 | 不使用 | 不使用 |
| 归一化、张量转换及批次补边 | 执行 | 执行 |

关闭 `multi_scale` 后，训练阶段同样使用 `fix_scale_size=512`，但仍保留训练时的随机翻转和颜色增强。随机增强按图片采样，不要求同批所有图片都应用相同增强。

上述尺寸来自当前的 [模型配置](src/configs/model.yaml)，不是 `AlbTransform` 类独立使用时的默认尺寸。相关配置片段为：

```yaml
model:
  patch_size: 16
train_config:
  multi_scale: true
  scale_sizes: [224, 512, 768]
  fix_scale_size: 512
```

### 等比例缩放与批次补边

缩放使每张图片的最长边达到选定尺寸，另一条边按比例计算，保留物体形状。例如原图为 `1920 × 1080`，最长边选择 `512` 后，得到 `512 × 288`。本节尺寸均按“宽 × 高”表示。

同批图片即使最长边相同，长宽比不同时，缩放后的宽高仍可能不同。因此先取批次最大高、最大宽，再分别向上补齐到 `model.patch_size` 的整数倍：

```text
H_pad = ceil(批次最大高度 / patch_size) × patch_size
W_pad = ceil(批次最大宽度 / patch_size) × patch_size
```

当前 `patch_size=16`；已能整除的尺寸保持不变。图片放在画布左上角，只在右侧和底部补边，填充值为归一化后的 `0`。补边不改变图片内容的长宽比；公共画布可能是长方形，也可能在混合横图、竖图时成为正方形。

同时生成形状为 `[B, H_pad, W_pad]` 的布尔掩码 `padding_masks`：`False` 表示有效图片区域，`True` 表示补边区域，供后续模型区分两者。掩码根据图片尺寸生成，与像素是否为黑色无关。

### 归一化与标注处理

RGB 像素使用 ImageNet 参数归一化，即 `(pixel / 255 - mean) / std`：

- `mean = (0.485, 0.456, 0.406)`。
- `std = (0.229, 0.224, 0.225)`。

边界框采用绝对像素 `xyxy` 格式，即 `[xmin, ymin, xmax, ymax]`，与图片同步缩放和翻转。输入框超出图片范围时先限制到图片边界，退化框会报错。右下补边不改变坐标原点，因此无需平移边界框；当前输出的框仍为缩放后图片上的像素坐标。

批次输出约定如下：

| 字段 | 类型与形状 | 含义 |
|---|---|---|
| `images` | `float32 [B, 3, H_pad, W_pad]` | 归一化并补边后的图片 |
| `boxes` | `list[float32 Tensor]`，每张为 `[N_i, 4]` | 按图片保存的变长边界框 |
| `labels` | `list[int64 Tensor]`，每张为 `[N_i]` | 与边界框顺序一致的类别编号 |
| `padding_masks` | `bool [B, H_pad, W_pad]` | 有效区域和补边区域的标记 |
| `origins` | `int64 [B, 2]` | 原始 `[width, height]`，用于后续坐标还原 |
| `image_ids` | `list[str]` | 原始图片编号 |

实现位置：[单图转换 AlbTransform](src/datasets/transforms.py)、[批次组装 RAODCollate](src/datasets/collate.py)、[批次类型 RAODBatch](src/datasets/types.py)。`transforms.py` 的预览测试会将增强概率临时设为 `1`，便于观察效果；实际数据加载使用上表中的默认概率。
