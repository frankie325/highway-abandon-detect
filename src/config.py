from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ModelConfig(_ConfigModel):
    """模型配置。"""

    pretrain_model: str = Field(min_length=1)
    patch_size: int = Field(gt=0)
    out_indices: list[int] = Field(min_length=1)
    projector_scale: list[Literal["P3", "P4", "P5", "P6"]]
    num_classes: int = Field(gt=0)
    two_stage: bool = True
    bbox_reparam: bool = True
    d_model: int = Field(gt=0)
    aux_loss:  bool = True
    num_queries: int = Field(gt=0)
    num_decoder_layers: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    dropout: float = Field(gt=0, lt=1)
    group_detr: int = Field(gt=0)
    sa_nheads: int = Field(gt=0)
    ca_nheads: int = Field(gt=0)


class TrainConfig(_ConfigModel):
    """训练配置。"""

    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    optimizer: str = Field(min_length=1)
    loss: str = Field(min_length=1)
    multi_scale: bool = True
    scale_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [1024, 1152, 1280], min_length=1
    )
    fix_scale_size: int = Field(default=1280, gt=0)
    cost_class: float = Field(default=1.0, gt=0)
    cost_bbox: float = Field(default=5.0, gt=0)
    cost_giou: float = Field(default=2.0, gt=0)
    focal_alpha: float = Field(default=0.25, gt=0)


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
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    seed: int = Field(lt=2**32)  # 随机种子
    model: ModelConfig
    train_config: TrainConfig
    logger: LoggerConfig = Field(default_factory=LoggerConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """读取 YAML、解析插值并校验配置，默认加载 src/configs/model.yaml。"""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "model.yaml"
    config = OmegaConf.load(config_path)
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return AppConfig.model_validate(data)
