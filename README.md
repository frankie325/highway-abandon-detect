# Highway Abandon Detect

高速公路抛洒物检测实验项目。

终端日志统一使用 Loguru，使用方法见 [Loguru 完整使用指南](docs/loguru.md)。

日志配置位于 `src/configs/index.yaml` 的 `logger` 节点：终端默认 INFO、文件默认 DEBUG，文件按时间命名写入项目根目录的 `logs/`，支持颜色、轮转、保留期限和压缩。

## 前置准备

### Hugging Face 模型授权

默认 DINOv3 权重来自受控模型 [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。首次训练前需要：

1. 登录 Hugging Face，打开模型页面，点击 **Agree and access repository** 或 **Request access**，确保当前账号已获得访问权限。
2. 在 [Access Tokens](https://huggingface.co/settings/tokens) 创建具有 `read` 权限的 token；token 必须属于已获得模型访问权限的账号。
3. 在项目环境中登录（凭据由 Hugging Face CLI 管理）：

```bash
uv run hf auth login
# 旧版本 huggingface_hub 使用：
uv run huggingface-cli login
```

按提示粘贴 token。不要将 token 写入代码或提交到 Git。可通过以下命令验证：

```bash
uv run python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('facebook/dinov3-vitb16-pretrain-lvd1689m', 'config.json'))"
```

如果仍提示 `Access to model ... is restricted`，请检查 token 所属账号是否已经接受访问条款；必要时先执行 `uv run hf auth logout`，再重新登录。模型 ID 来自 `src/configs/haod_detr_base.yaml` 的 `pretrain_model`；获得权重后也可改为本地模型目录。

### SwanLab 登录

训练通过 SwanLab 云端记录实验。首次使用需登录：

```bash
uv run swanlab login
```

按提示在浏览器完成登录或粘贴 API Key。凭据由 SwanLab CLI 管理，不写入配置文件，也不需要 `.env`。

## 训练与实验记录

完成上述前置准备后启动：

```bash
uv sync
uv run python train.py
```

项目总配置固定为 [src/configs/index.yaml](src/configs/index.yaml)，包括训练、日志和 SwanLab 设置。总配置中的 `model_config` 指向 [haod_detr_base.yaml](src/configs/haod_detr_base.yaml)，路径相对于总配置文件所在目录。

优化器可设置为 `SGD`、`Adam` 或 `AdamW`，也可通过 `--optimizer` 覆盖。三者共用 `lr` 和 `weight_decay`；`momentum` 只用于 SGD，默认值为 0.9。`weight_decay` 默认是 0.0，以保持现有 Adam 行为；如需 AdamW 的权重衰减效果，应设置大于 0 的值。恢复检查点会加载其中的优化器状态，包括参数组中的超参数；切换优化器或希望使用新的学习率、权重衰减时应从头训练。

优化器参考 RF-DETR 使用分组学习率：检测头和投影层为 `lr`，Decoder 为 `lr × lr_component_decay`；DINOv3 编码器默认冻结，不参与优化。要微调 DINOv3，在 `haod_detr_base.yaml` 中设置 `freeze_encoder: false`；其各层按 `lr_encoder × lr_component_decay² × lr_vit_layer_decay^(N+1-layer_id)` 设置学习率，其中 embeddings 的层号为 0、第 i 个 Transformer 层为 i+1、输出 norm 为 N+1，N 是实际编码器层数。编码器的 bias、norm 和 embeddings 等参数不使用权重衰减。

分层学习率改变了优化器参数组，启用前的单组优化器检查点无法直接用于完整断点续训；冻结状态和分层学习率设置也应在续训时保持一致。解冻 DINOv3 会增加训练所需计算和内存。

学习率策略由 `train_config.lr_scheduler` 选择，也可用 `--lr-scheduler` 覆盖：

- `cosine`（默认）：按优化器更新步数执行线性 Warmup 和余弦衰减。`warmup_ratio: 0.05` 表示前约 5% 的更新步数预热，`min_lr_ratio: 0.01` 表示结束时各组衰减至各自最大学习率的 1%；`warmup_ratio: 0.0` 可关闭预热。
- `step`：不做 Warmup；每完成 `step_size` 个 epoch，将各组学习率乘以 `step_gamma`。例如 `step_size: 30`、`step_gamma: 0.1` 表示每 30 轮降为原来的十分之一。

两种策略都保留分组间的学习率比例，并由现有 SwanLab 记录学习率。

检查点会保存优化器和调度器进度。断点续训应使用相同的调度器类型、训练总轮数、数据及调度参数；改变这些设置可能改变曲线或与旧状态不兼容。加入调度器之前保存的旧检查点没有调度器状态，无法直接作为完整断点恢复；需要从头训练或另行进行权重迁移。

SwanLab 项目和实验名在总配置的 `swanlab` 节点设置。每次启动时，程序会在基础实验名后追加本地时间，例如 `haod-detr-training-2026-11-16_14-30-00-123456`。登录凭据由 `uv run swanlab login` 管理，不写入配置文件。项目不再加载 `.env` 文件或读取 SwanLab 配置环境变量。

每轮正式验证时，SwanLab 的 `val/predictions` 会保存首个验证批次中每张图片的预览。图片中绿色为真实框、红色为预测框、紫色为 padding；即使没有预测框达到阈值，图片也会保留。验证同时记录总损失和各最终层损失。

验证还会在整个验证集上计算 `val_map`（mAP@0.5:0.95）、`val_map_50` 和 `val_map_75`，并向 SwanLab 记录 `val/pr_curve` 与 `val/confusion_matrix`。PR 曲线采用 IoU=0.5 和 COCO 的 101 个召回率采样点；mAP 使用每张图最高分的 `evaluate_config.max_detections` 个预测，不做置信度阈值过滤。混淆矩阵的置信度和 IoU 阈值由 `evaluate_config` 配置，默认分别为 0.3 和 0.5；预测按分数排序并与真实框按 IoU 一对一匹配，行是真实类别、列是预测类别，背景行表示误检、背景列表示漏检，背景/背景不计真负例。 验证预览单独使用 `preview_score_threshold` 和 `preview_max_predictions`。

评估框坐标基于缩放后图片的有效区域，排除 padding；当前管线不保留原图面积和 crowd 标注。验证集中没有可评估真实目标时，COCO 指标可能为 `-1`，表示不可用而非负准确率。图表由 SwanLab 保存到输出目录内的实验媒体目录。

默认输出结构（目录按需创建）：

```text
logs/                 # Loguru 的时间命名日志文件
outputs/
├── checkpoints/
│   ├── best.ckpt     # val_map 最高的检查点，用于独立验证
│   └── last.ckpt     # 最近一次检查点，用于断点续训
├── swanlab/          # SwanLab 本地实验数据和媒体文件
└── predictions/      # 每次预测单独保存一组原图预览
```

`outputs/` 不纳入 Git；已有旧目录不会自动迁移。日志目录可在配置中调整，`--outputs` 可指定检查点和 SwanLab 的输出目录。

```bash
# 覆盖常用训练参数；未指定的值沿用 YAML 配置。
uv run python train.py --batch-size 2 --epochs 100 --learning-rate 0.001
# 使用 SGD 和动量（此处显式覆盖配置中的权重衰减）。
uv run python train.py --optimizer SGD --momentum 0.9 --weight-decay 0.0001
# 使用按 epoch 衰减的 StepLR。
uv run python train.py --lr-scheduler step
# 从 outputs/checkpoints/last.ckpt 恢复训练。
uv run python train.py --resume
# 使用指定输出目录，验证其中的 checkpoints/best.ckpt。
uv run python eval.py --outputs outputs
# 查看所有参数；两者均支持 --batch-size、--num-workers 和 --seed。
uv run python train.py --help
uv run python eval.py --help
```

`--outputs` 的相对路径以项目根目录为基准，默认是 `outputs`。在 YAML 中设置 `train_config.resume: true` 或传入 `--resume` 时，从输出目录下的 `checkpoints/last.ckpt` 恢复；默认从头训练。独立验证读取 `checkpoints/best.ckpt`，需要提供与训练时模型结构一致的配置。旧实验按 `val_loss` 选出的检查点需要重新评估，才能按 `val_map` 比较。

训练和验证的数据集由总配置中的 `train_config.dataset` 指定。默认的 `raod_small` 使用固定种子的 80/20 训练和验证划分。选择 `raod` 时，训练使用完整数据集的官方 `train`，验证使用官方 `test`。命令行可以覆盖 YAML，例如：

```bash
uv run python train.py --dataset raod --outputs outputs/raod
uv run python eval.py --dataset raod --outputs outputs/raod
```

## 图片预测

```bash
uv run python predict.py road.jpg
uv run python predict.py road1.jpg road2.png --outputs outputs/run1 --batch-size 2 --score-threshold 0.4
```

图片路径相对于当前工作目录。`--outputs` 沿用训练的路径规则，预测加载其中的 `checkpoints/best.ckpt`，模型结构须与训练时一致。`predict_config` 设置独立预测的批次大小、最高分候选框数量和置信度阈值，默认分别为 `1`、`100` 和 `0.3`。`--batch-size` 和 `--score-threshold` 可覆盖 YAML。预测复用验证时的图像预处理，无需 RAOD 数据集或 SwanLab 登录。

Lightning 的 `Trainer.predict()` 按批运行推理，每批完成后保存结果。每次运行在 `predictions/` 下创建独立的 `run-*` 目录，按输入顺序保存 `0001_road.png` 等文件。预测框还原到原图坐标，并标注类别和置信度，例如 `spill 0.92`。没有检测结果的图片也会保存。

## 数据集图像变换策略

当前使用 Albumentations 处理单张图片及边界框，由 `Collator` 选择缩放尺寸、完成补边并生成掩码，同时保留原始样本。处理流程为：

```text
PIL 图片转 RGB → 等比例缩放 → 训练时随机增强 → 归一化 → CHW 张量 → 批次右下补边
```

### 训练与验证

| 处理项 | 训练阶段 | 验证阶段 |
|---|---|---|
| 缩放策略 | 开启多尺度时，每个 batch 从 `[768, 832, 896, 960, 1024, 1088, 1152]` 中随机选择一个最长边尺寸 | 最长边固定为 `960` |
| 长宽比 | 保持原始比例，同批图片共享目标最长边 | 保持原始比例 |
| 水平翻转 | 每张图片以 `0.5` 的概率执行，边界框同步翻转 | 关闭 |
| 亮度与对比度 | 每张图片以 `0.3` 的概率执行，两个变化范围参数均为 `(-0.1, 0.1)` | 关闭 |
| 图像裁剪 | 不使用 | 不使用 |
| 归一化、张量转换及批次补边 | 执行 | 执行 |

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

| 字段 | 类型与形状 | 含义 |
|---|---|---|
| `NestedTensor.tensors` | `float32 [B, 3, H_pad, W_pad]` | 归一化并补边后的图片 |
| `NestedTensor.mask` | `bool [B, H_pad, W_pad]` | True 表示补边 |
| `targets[i]["boxes"]` | `float32 [N_i, 4]` | 第 i 张图的归一化 `cxcywh` 框 |
| `targets[i]["labels"]` | `int64 [N_i]` | 与边界框顺序一致的类别编号 |

实现位置：[单图转换 AlbTransform](src/datasets/transforms.py)、[批次组装 Collator](src/datasets/collator.py)、[批次类型 DetectionBatch](src/datasets/types.py)。`transforms.py` 的预览测试会将增强概率临时设为 `1`，便于观察效果；实际数据加载使用上表中的默认概率。
