"""项目终端及文件日志配置：入口配置一次，其他模块直接使用 logger。"""

import sys
import tomllib
from pathlib import Path

from ..config import LoggerConfig
from loguru import logger
from tqdm import tqdm


# 日志布局统一在此维护：项目名称 | 时间 | 等级 | 模块:函数:行号 | 消息。
# 终端使用颜色标签，文件使用纯文本；不作为实验配置项暴露。
CONSOLE_FORMAT = (
    "<cyan>[{extra[project_name]}]</cyan> | "
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>"
)
FILE_FORMAT = (
    "[{extra[project_name]}] | {time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level:<8} | {name}:{function}:{line} | {message}"
)


def setup_logging(config: LoggerConfig) -> None:
    """初始化终端及时间命名的文件日志，终端通过 tqdm.write 配合进度条。"""
    project_root = Path(__file__).resolve().parents[2]
    # 项目名以包元数据为唯一来源，每次初始化读取，不受命令执行目录影响。
    with (project_root / "pyproject.toml").open("rb") as file:
        project_name = tomllib.load(file)["project"]["name"]

    log_path = None
    if config.file_enabled:
        # 相对路径始终以项目根目录为基准，不受命令执行目录影响。
        log_dir = Path(config.log_dir).expanduser()
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        # 保留 time 占位符，让 Loguru 的历史文件匹配规则覆盖之前的运行。
        # 微秒时间戳用于区分文件；进程号可按需显示在消息格式中。
        log_path = log_dir / "{time:YYYY-MM-DD_HH-mm-ss_SSSSSS}.log"

    # 移除默认或上次配置的输出，避免同一条日志被重复打印。
    logger.remove()
    # 全局默认字段使每个模块直接导入的 logger 都能输出项目名称。
    logger.configure(extra={"project_name": project_name})
    if config.console_enabled:
        logger.add(
            lambda message: tqdm.write(message, end="", file=sys.stderr),
            level=config.level,
            format=CONSOLE_FORMAT,
            colorize=config.colorize and sys.stderr.isatty(),
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
        )
    if log_path is not None:
        logger.add(
            log_path,
            level=config.file_level,
            format=FILE_FORMAT,
            colorize=False,  # 文件不写入终端 ANSI 颜色控制符。
            encoding="utf-8",
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            serialize=config.serialize,
        )
