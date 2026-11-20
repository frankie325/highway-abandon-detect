from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


type RAODDatasetName = Literal["raod_small", "raod"]


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ModelConfig(_ConfigModel):
    """模型配置。"""

    pretrain_model: str = Field(min_length=1)
    freeze_encoder: bool = True
    patch_size: int = Field(gt=0)
    out_indices: list[int] = Field(min_length=1)
    projector_scale: list[Literal["P3", "P4", "P5", "P6"]]
    num_classes: int = Field(gt=0)
    two_stage: bool = True
    bbox_reparam: bool = True
    d_model: int = Field(gt=0)
    aux_loss: bool = True
    num_queries: int = Field(gt=0)
    num_decoder_layers: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    dropout: float = Field(gt=0, lt=1)
    group_detr: int = Field(gt=0)
    dec_n_points: int = Field(gt=0)
    sa_nheads: int = Field(gt=0)
    ca_nheads: int = Field(gt=0)


class LossWeightsConfig(_ConfigModel):
    """检测损失权重，辅助层和编码器使用相同权重。"""

    loss_ce: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    loss_bbox: float = Field(default=5.0, ge=0, allow_inf_nan=False)
    loss_giou: float = Field(default=2.0, ge=0, allow_inf_nan=False)


class TrainConfig(_ConfigModel):
    """训练配置。"""

    dataset: RAODDatasetName = "raod_small"
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    lr: float = Field(gt=0, allow_inf_nan=False)
    lr_encoder: float = Field(default=0.00015, gt=0, allow_inf_nan=False)
    lr_vit_layer_decay: float = Field(default=0.8, gt=0, le=1, allow_inf_nan=False)
    lr_component_decay: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False)
    optimizer: Literal["sgd", "adam", "adamw"]
    momentum: float = Field(default=0.9, ge=0, lt=1, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    lr_scheduler: Literal["step", "cosine"] = "cosine"
    step_size: int = Field(default=30, gt=0)
    step_gamma: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1, allow_inf_nan=False)
    min_lr_ratio: float = Field(default=0.01, ge=0, le=1, allow_inf_nan=False)
    loss: Literal["ia_bce"] = "ia_bce"
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)
    resume: bool = False
    multi_scale: bool = True
    scale_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [1024, 1152, 1280], min_length=1
    )
    fix_scale_size: int = Field(default=1280, gt=0)
    cost_class: float = Field(default=1.0, gt=0)
    cost_bbox: float = Field(default=5.0, gt=0)
    cost_giou: float = Field(default=2.0, gt=0)
    focal_alpha: float = Field(default=0.25, gt=0)

    @field_validator("optimizer", mode="before")
    @classmethod
    def normalize_optimizer(cls, value: object) -> object:
        """配置中允许使用 SGD、Adam、AdamW 等常见大小写写法。"""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("lr_scheduler", mode="before")
    @classmethod
    def normalize_lr_scheduler(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class EvaluateConfig(_ConfigModel):
    """验证指标、混淆矩阵和验证预览参数。"""

    max_detections: int = Field(default=100, gt=0)
    confusion_score_threshold: float = Field(
        default=0.3, ge=0, le=1, allow_inf_nan=False
    )
    confusion_iou_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    preview_score_threshold: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    preview_max_predictions: int = Field(default=20, gt=0)


class PredictConfig(_ConfigModel):
    """独立图片预测参数。"""

    images: list[str] = Field(default_factory=list)
    batch_size: int = Field(default=1, gt=0)
    max_detections: int = Field(default=100, gt=0)
    score_threshold: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)


class SwanLabConfig(_ConfigModel):
    """实验记录配置，登录凭据由 SwanLab 管理。"""

    project: str = Field(default="highway-abandon-detect", min_length=1)
    experiment_name: str = Field(default="haod-detr-training", min_length=1)


class LoggerConfig(_ConfigModel):
    """终端和文件日志配置，文件路径相对于项目根目录。"""

    # 终端最低输出级别：INFO 表示显示 INFO 及以上，隐藏 DEBUG 和 TRACE。
    level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"
    # 文件最低输出级别：默认 DEBUG，比终端保留更多调试信息。
    file_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "DEBUG"
    # 是否在终端显示日志；关闭后不影响文件输出。
    console_enabled: bool = True
    # 是否保存日志文件；关闭后不会创建日志目录或文件。
    file_enabled: bool = True
    # 日志保存目录：相对路径以项目根目录为基准，也支持绝对路径。
    log_dir: str = Field(default="logs", min_length=1)
    # 是否美化终端颜色；仅在终端支持颜色时生效，文件始终不写入颜色控制符。
    colorize: bool = True
    # 文件轮转条件：如 "50 MB" 或 "1 day"；None 表示不轮转。
    # 满足条件后关闭当前文件，后续日志写入新文件。
    rotation: str | None = Field(default="50 MB", min_length=1)
    # 历史文件最长保留时间；None 表示不按期限清理。
    # 清理在轮转时触发；未配置轮转时在处理器关闭时触发，不是定时清理任务。
    retention: str | None = Field(default="14 days", min_length=1)
    # 历史日志文件的归档压缩格式；None 表示保留未压缩文件。
    compression: (
        Literal["zip", "gz", "bz2", "xz", "lzma", "tar", "tar.gz", "tar.bz2", "tar.xz"]
        | None
    ) = "zip"
    # True：先将日志放入队列，由后台线程写入，减少日志 I/O 对训练的阻塞。
    # False：在调用 logger 的线程中直接写入；使用队列时可用 logger.complete() 等待写完。
    enqueue: bool = True
    # 异常发生时，是否补充捕获位置以上的调用链，帮助追踪错误从哪里进入。
    backtrace: bool = True
    # 是否在异常堆栈中展开变量值；默认关闭，避免输出凭据或大量张量内容。
    diagnose: bool = False
    # 仅影响文件输出：True 为逐行 JSON，便于程序解析；False 为可读文本。
    serialize: bool = False

    @field_validator("level", "file_level", mode="before")
    @classmethod
    def normalize_level(cls, value: object) -> object:
        """允许 YAML 使用 info、debug 等小写级别。"""
        return value.strip().upper() if isinstance(value, str) else value


class AppConfig(_ConfigModel):
    """实验配置，基础训练字段必填，预处理和日志参数提供默认值。"""

    # 运行目录默认指向项目根目录
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    outputs: str = Field(default="outputs", min_length=1)
    seed: int = Field(lt=2**32)  # 随机种子
    model: ModelConfig
    train_config: TrainConfig
    evaluate_config: EvaluateConfig = Field(default_factory=EvaluateConfig)
    predict_config: PredictConfig = Field(default_factory=PredictConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    swanlab: SwanLabConfig = Field(default_factory=SwanLabConfig)


def load_config(
    config_path: str | Path | None = None, *, overrides: dict | None = None
) -> AppConfig:
    """读取总配置及其引用的模型配置，合并覆盖项后解析插值并校验。"""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "index.yaml"
    config_path = Path(config_path).expanduser().resolve()
    config = OmegaConf.load(config_path)
    model_file = config.get("model_config")
    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError("总配置中的 model_config 必须指定模型 YAML 文件路径")
    model_path = Path(model_file).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"模型配置文件不存在：{model_path}")
    del config["model_config"]
    config.model = OmegaConf.load(model_path)
    config = OmegaConf.merge(config, overrides or {})
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return AppConfig.model_validate(data)
