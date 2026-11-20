# Highway Abandon Detect

高速公路抛洒物检测实验项目。

## 前置准备

### 环境安装

1. 安装 `uv`。macOS 或 Linux 可以使用官方安装脚本，Windows 可以在 PowerShell 中运行 `winget install --id=astral-sh.uv -e`。安装完成后，新开终端运行 `uv --version` 检查。[uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/)

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. 进入项目根目录（包含 `pyproject.toml` 和 `uv.lock`），运行以下命令，安装依赖

   ```bash
   uv sync
   ```

3. 检查 Python、PyTorch 和加速设备。后续使用 `uv run` 执行命令即可在项目环境中运行。

   ```bash
   uv run --locked python --version
   uv run --locked python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
   ```

   项目在 macOS 使用 PyTorch 的 CPU 软件源，支持 MPS 的 Mac 可自动使用 MPS。其他平台配置为 CUDA 12.6 软件源。使用 NVIDIA GPU 时，需安装兼容的 NVIDIA 驱动，并确认上面的检查输出 `CUDA: True`。

### 登陆Hugging Face

```bash
uv run hf auth login
# 旧版本 huggingface_hub 使用：
uv run huggingface-cli login
```

### Hugging Face 模型授权

默认 DINOv3 权重来自受控模型 [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。首次训练前需要：

1. 打开模型页面，点击 **Agree and access repository** 或 **Request access**，确保当前账号已获得访问权限。
2. 等待授权通过

### SwanLab 登录

训练通过 SwanLab 云端记录实验。首次使用需登录：

```bash
uv run swanlab login
```

按提示在浏览器完成登录或粘贴 API Key。凭据由 SwanLab CLI 管理，不写入配置文件，也不需要 `.env`。

## 训练、验证与预测

在项目根目录运行以下命令。三个脚本均支持 `--outputs`，用于覆盖 [总配置](src/configs/index.yaml) 中的 `outputs`。相对路径以项目根目录为基准，命令行未指定的选项沿用 YAML。

### 训练

```bash
# 按 YAML 配置训练
uv run python train.py
# 覆盖批次大小、轮数和学习率
uv run python train.py --batch-size 2 --epochs 100 --lr 0.001
# 使用 SGD 和动量
uv run python train.py --optimizer SGD --momentum 0.9 --weight-decay 0.0001
# 使用按轮衰减的 StepLR
uv run python train.py --lr-scheduler step
# 使用完整 RAOD 数据集，结果保存到独立目录
uv run python train.py --dataset raod --outputs outputs/raod
# 从同一输出目录的 last.ckpt 续训
uv run python train.py --resume
```

训练和验证共用 `train_config.dataset`。默认的 `raod_small` 使用固定种子的 80/20 划分。完整 `raod` 使用官方 `train` 训练，使用官方 `test` 验证。

续训时指定原来的 `--outputs` 和训练配置。`--resume` 或 YAML 中的 `train_config.resume: true` 会读取对应目录下的 `checkpoints/last.ckpt`。

### 独立验证

```bash
# 验证默认输出目录里的最佳检查点
uv run python eval.py
# 验证指定目录中的检查点，使用完整 RAOD 数据集
uv run python eval.py --dataset raod --outputs outputs/raod
# 调整每张图参与评估的候选框上限
uv run python eval.py --max-detections 75
```

验证会加载 `<outputs>/checkpoints/best.ckpt`。模型结构须与检查点一致，复现实验结果时也应选用原来的数据集。每次独立验证都会在 SwanLab 创建一个新实验，记录损失、检测指标和验证预览。混淆矩阵阈值和验证预览设置在 YAML 的 `evaluate_config` 中调整，命令行选项见 `eval.py --help`。

### 图片预测

```bash
# 预测单张图片
uv run python predict.py road.jpg
# 对多张图片分批预测，并指定模型所在的输出目录
uv run python predict.py road1.jpg road2.png --outputs outputs/raod --batch-size 2 --score-threshold 0.4
```

图片路径相对于当前工作目录。预测加载 `<outputs>/checkpoints/best.ckpt`，模型结构须与训练时一致。预测复用验证的图像预处理，无需 RAOD 数据集或 SwanLab 登录。`predict_config` 中的批次大小、候选框数量和置信度阈值默认分别为 `1`、`100` 和 `0.3`。命令行可用 `--batch-size` 和 `--score-threshold` 覆盖对应设置。

结果逐批按输入顺序保存到 `<outputs>/predictions/run-*/`，文件名如 `0001_road.png`。预测框绘制在原始尺寸的图片上，标签示例为 `spill 0.92`。没有检测结果的图片也会保存。

### 查看可用参数

```bash
uv run python train.py --help
uv run python eval.py --help
uv run python predict.py --help
```

## 项目配置

总配置位于 [index.yaml](src/configs/index.yaml)，各参数的默认值和说明见 YAML 注释。

| 配置项            | 用途                                                                                                      |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| `seed`            | 随机种子                                                                                                  |
| `outputs`         | 输出目录，默认是 `outputs/`                                                                             |
| `model_config`    | 引用模型配置，默认是 [haod_detr_base.yaml](src/configs/haod_detr_base.yaml)，包含 DINOv3 权重和检测器结构 |
| `train_config`    | 数据集、训练轮数、批次大小、优化器、学习率策略和续训                                                      |
| `evaluate_config` | mAP 的候选框上限、混淆矩阵阈值和验证预览设置                                                              |
| `predict_config`  | 图片路径、预测批次大小、候选框上限和置信度阈值                                                            |
| `logger`          | 终端和文件日志设置，文件默认写入 `logs/`                                                                  |
| `swanlab`         | 云端实验的项目名和实验名                                                                                  |

检查点、SwanLab 本地记录和预测图片分别保存在输出目录的 `checkpoints/`、`swanlab/` 和 `predictions/`。

## 数据集图像变换策略

当前使用 Albumentations 处理单张图片及边界框，由 `Collator` 选择缩放尺寸、完成补边并生成掩码，同时保留原始样本。处理流程为：

```text
PIL 图片转 RGB → 等比例缩放 → 训练时随机增强 → 归一化 → CHW 张量 → 批次右下补边
```

### 训练与验证

| 处理项                     | 训练阶段                                                                                      | 验证阶段           |
| -------------------------- | --------------------------------------------------------------------------------------------- | ------------------ |
| 缩放策略                   | 开启多尺度时，每个 batch 从 `[768, 832, 896, 960, 1024, 1088, 1152]` 中随机选择一个最长边尺寸 | 最长边固定为 `960` |
| 长宽比                     | 保持原始比例，同批图片共享目标最长边                                                          | 保持原始比例       |
| 水平翻转                   | 每张图片以 `0.5` 的概率执行，边界框同步翻转                                                   | 关闭               |
| 亮度与对比度               | 每张图片以 `0.3` 的概率执行，两个变化范围参数均为 `(-0.1, 0.1)`                               | 关闭               |
| 图像裁剪                   | 不使用                                                                                        | 不使用             |
| 归一化、张量转换及批次补边 | 执行                                                                                          | 执行               |

关闭 `multi_scale` 后，训练阶段同样使用 `fix_scale_size=960`，但仍保留训练时的随机翻转和颜色增强。随机增强按图片采样，不要求同批所有图片都应用相同增强。

上述尺寸来自当前的 [总配置](src/configs/index.yaml)，不是 `AlbTransform` 类独立使用时的默认尺寸。相关配置片段为：

```yaml
model:
  patch_size: 16
train_config:
  multi_scale: true
  scale_sizes: [768, 832, 896, 960, 1024, 1088, 1152]
  fix_scale_size: 960
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

输入边界框采用像素 `xyxy` 格式，与图片同步缩放和翻转。转换后，以变换后的有效图片宽高归一化，并转换为 `cxcywh`。右下补边不改变坐标原点；损失和模型均使用相对于有效图片的归一化框。

批次输出为 `(NestedTensor, targets)`：

| 字段                   | 类型与形状                     | 含义                          |
| ---------------------- | ------------------------------ | ----------------------------- |
| `NestedTensor.tensors` | `float32 [B, 3, H_pad, W_pad]` | 归一化并补边后的图片          |
| `NestedTensor.mask`    | `bool [B, H_pad, W_pad]`       | True 表示补边                 |
| `targets[i]["boxes"]`  | `float32 [N_i, 4]`             | 第 i 张图的归一化 `cxcywh` 框 |
| `targets[i]["labels"]` | `int64 [N_i]`                  | 与边界框顺序一致的类别编号    |

实现位置：[单图转换 AlbTransform](src/datasets/transforms.py)、[批次组装 Collator](src/datasets/collator.py)、[批次类型 DetectionBatch](src/datasets/types.py)。`transforms.py` 的预览测试会将增强概率临时设为 `1`，便于观察效果；实际数据加载使用上表中的默认概率。

## Todo

resume时继续从上一次swanlab实验恢复
预测脚本输出目录名称修改
