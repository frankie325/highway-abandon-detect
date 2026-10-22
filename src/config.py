from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ModelConfig(_ConfigModel):
    """模型配置。"""

    seed: int = Field(lt=2**32)
    name: str = Field(min_length=1)
    patch_size: int = Field(gt=0)
    out_indices: list[int] = Field(min_length=1)


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


class AppConfig(_ConfigModel):
    """实验配置，基础训练字段必填，预处理参数提供默认值。"""

    model: ModelConfig
    train_config: TrainConfig


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """读取 YAML、解析插值并校验配置，默认加载 src/configs/model.yaml。"""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "model.yaml"
    config = OmegaConf.load(config_path)
    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return AppConfig.model_validate(data)
