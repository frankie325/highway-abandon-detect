# 仓库贡献指南

## 项目结构

本项目使用以 DINOv3 为骨干网络的 DETR 模型检测高速公路抛洒物。请在仓库根目录运行 `train.py`、`eval.py` 和 `predict.py`。

- `src/models/` 存放检测器和骨干网络，`src/loss/` 存放匹配与损失计算。
- `src/datasets/` 负责加载 RAOD 数据和处理图像，`src/trainer/` 包含 Lightning 训练、评估指标和预测逻辑。
- 在 `src/configs/index.yaml` 修改实验默认值，在 `src/configs/haod_detr_base.yaml` 修改模型设置。环境配置见 `README.md`，实现说明见 `docs/`。
- Git 忽略 `data/`、`cache/`、`logs/` 和 `outputs/`。

## 开发命令

- `uv sync --locked` 在项目的 Python 3.12 环境中安装锁定版本的依赖。
- `uv run python train.py --dataset raod_small --outputs outputs/dev` 使用默认的 RAOD 子集训练，结果写入独立目录。DINOv3 权重授权和 SwanLab 登录方式见 `README.md`。
- `uv run python eval.py --outputs outputs/dev` 加载该次训练生成的 `checkpoints/best.ckpt` 进行评估。数据集和模型配置应与训练时一致。
- `uv run python predict.py road.jpg --outputs outputs/dev` 使用同一检查点预测图片，将带检测框的结果保存至 `outputs/dev/predictions/`。
- 在任一脚本命令后加 `--help` 可查看可用参数。

## 编码风格

Python 使用 4 个空格缩进。函数、模块和 YAML 配置键使用 `snake_case`，类名使用 `PascalCase`。类型注解遵循相邻代码的写法。跨模块传递数据时，说明张量形状、检测框坐标格式和掩码含义。仓库目前未配置格式化工具或代码检查工具。

## 测试

使用 `unittest`，测试文件命名为 `test_*.py`，测试方法以 `test_` 开头。本地有测试文件时，运行 `uv run python -m unittest discover -s tests -v`。Git 忽略 `tests/`，因此新克隆的仓库中没有这些测试。回归测试可使用小批量合成数据或模拟骨干网络。仓库目前未设定覆盖率门槛。

## 提交与拉取请求

沿用 Git 历史中的简短提交标题，如 `feat: 添加预测脚本`。其他常用前缀包括 `fix:`、`refactor:`、`doc:` 和 `chore:`。在 PR 中说明行为变化，链接相关议题，并列出验证命令。涉及实验变更时，附上数据集、覆盖的配置项，以及相关指标或预测预览。数据集、模型权重、日志和凭据仅保存在本地。

## 代码导航

CodeGraph 可用时，用 `codegraph_context` 了解模块结构，用 `codegraph_search` 查找符号，用 `codegraph_trace` 追踪调用路径。查找文本时使用 `rg`。如果索引尚未初始化，先确认是否要运行 `codegraph init -i`。
