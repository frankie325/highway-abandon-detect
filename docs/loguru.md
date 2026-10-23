# Loguru 完整使用指南

本文对应项目已安装的 **Loguru 0.7.3**，按官方概览的 17 项现行特性逐项说明，并补充公开 API、输出参数和项目用法。示例根据本地版本的签名与行为编写，参考入口见文末。

除特别标注的片段外，下面的 Python 示例默认已执行：

```python
import sys
from loguru import logger
```

每段按独立示例理解。`logger.remove()` 会移除当前全部输出目标，适合在入口重新配置，不能每个 batch 都调用。文件示例会在运行目录创建 `logs/`，通知示例需要额外依赖和自己的服务地址。

## 目录与特性总览

| 序号 | 官方特性 | 主要入口 |
|---|---|---|
| 1 | 导入即可使用 | 全局 `logger` |
| 2 | 统一配置输出、格式和过滤 | `add()`、`remove()` |
| 3 | 文件轮转、保留和压缩 | `rotation`、`retention`、`compression` |
| 4 | 花括号格式化 | `logger.info("{}", value)` |
| 5 | 捕获函数、线程入口中的异常 | `catch()` |
| 6 | 终端彩色输出 | `colorize`、颜色标签 |
| 7 | 线程、进程和异步协作 | `enqueue`、`context`、`complete()` |
| 8 | 详细异常堆栈 | `exception()`、`backtrace`、`diagnose` |
| 9 | 结构化日志和上下文 | `serialize`、`bind()`、`contextualize()`、`patch()` |
| 10 | 延迟计算日志参数 | `opt(lazy=True)` |
| 11 | 内置和自定义级别 | 级别方法、`level()`、`log()` |
| 12 | 时间和时区格式化 | `{time:...}` |
| 13 | 应用、脚本与库的配置方式 | `configure()`、`enable()`、`disable()` |
| 14 | 与标准库 logging 互通 | `logging.Handler`、转发处理器 |
| 15 | 环境变量定制默认行为 | `LOGURU_*` |
| 16 | 解析已有文本日志 | `parse()` |
| 17 | 通过第三方服务发送通知 | 自定义 sink、Apprise |

官方概览还保留了一项被划掉的“性能提升十倍”描述，本文将其视为历史宣传，不作为当前版本的性能保证。

## 0. 本项目怎么使用

项目日志配置函数位于 `src/utils/logger.py`。入口配置一次，业务模块直接导入同一个 `logger`：

```python
from loguru import logger
from src.config import load_config
from src.utils.logger import setup_logging

if __name__ == "__main__":
    config = load_config()
    setup_logging(config.logger)
    logger.info("开始读取 RAOD 数据")
    logger.complete()
```

该配置默认同时输出到终端 `stderr` 和项目根目录的 `logs/`。终端使用 `tqdm.write()` 配合进度条，默认 INFO；文件保留 DEBUG 及以上日志。文件名如 `2026-10-23_09-30-00_123456.log`，包含日期、时分秒与微秒。移动配置模块后，入口的导入路径也要同步，旧的 `from src.logger import setup_logging` 应改为上面的路径。

默认格式为“项目名称 | 时间 | 日志等级 | 模块:函数:行号 | 消息”，例如：

```text
[highway-abandon-detect] | 2026-10-23 09:30:00.123 | INFO     | src.engine:train:36 | 配置加载完成
```

终端对项目名、时间、等级及代码位置着色，重定向输出时不强制加入颜色；文件始终不包含 ANSI 颜色控制符。

### 项目的 LoggerConfig 配置项

配置位于 `src/configs/model.yaml` 的 `logger` 节点，经 Pydantic 校验后传给 `setup_logging()`。日志级别可使用大小写，例如 `info` 会转换为 `INFO`。

日志前缀中的项目名在初始化时自动从项目根目录 `pyproject.toml` 的 `[project].name` 读取，不在 `LoggerConfig` 或 YAML 中重复配置，也不依赖命令执行时的工作目录。

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `level` / `file_level` | `INFO` / `DEBUG` | 终端与文件各自的最低级别 |
| `console_enabled` / `file_enabled` | `true` / `true` | 独立开启或关闭两类输出 |
| `log_dir` | `logs` | 相对于项目根目录，也支持绝对路径 |
| `colorize` | `true` | 在支持颜色的终端中美化输出 |
| `rotation` | `50 MB` | 单文件大小达到阈值时轮转；`null` 为不轮转 |
| `retention` | `14 days` | 轮转时清理过期文件；`null` 为不按期限清理 |
| `compression` | `zip` | 历史文件归档格式；`null` 为不压缩 |
| `enqueue` | `true` | 队列写入，减少训练线程执行日志 I/O |
| `backtrace` / `diagnose` | `true` / `false` | 异常调用链与变量值展示开关 |
| `serialize` | `false` | 设为 true 后文件输出逐行 JSON，终端保持可读文本 |

没有轮转配置时，保留与压缩也会在处理器关闭时执行；设置了轮转时，它们在轮转时执行，不是后台定时清理。文件名保留 Loguru 的时间占位符，使历史运行的时间命名文件也能参与匹配。程序主动退出前调用 `logger.complete()`，等待已入队消息处理完毕。

需要调整级别时修改配置，或构造配置对象：

```python
from src.config import LoggerConfig
from src.utils.logger import setup_logging

setup_logging(LoggerConfig(level="DEBUG"))
```

### 三种格式模板

日志格式不放在 `LoggerConfig` 或 YAML 中，统一维护在 `src/utils/logger.py` 的 `CONSOLE_FORMAT`、`FILE_FORMAT` 常量中。下面的纯文本模板可作为修改 `FILE_FORMAT` 的参考；`CONSOLE_FORMAT` 可在字段外包上 `<cyan>`、`<green>`、`<level>` 等颜色标签。

**简洁格式：项目、时间、等级、简短位置和消息。**

```text
[{extra[project_name]}] | {time:HH:mm:ss} | {level:<8} | {module}:{line} | {message}
```

**详细格式：补充毫秒、进程、线程及完整代码位置。**

```text
[{extra[project_name]}] | {time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | pid={process.id} tid={thread.id} | {name}:{function}:{line} | {message}
```

**训练格式：补充 epoch 和 step，便于追踪实验。**

```text
[{extra[project_name]}] | {time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} | epoch={extra[epoch]} step={extra[step]} | {message}
```

训练格式要求每条消息都具备 epoch、step。入口先设置默认值，再按训练进度临时覆盖：

```python
from src.config import LoggerConfig
from src.utils.logger import setup_logging

training_format = (
    "[{extra[project_name]}] | {time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "{name}:{line} | epoch={extra[epoch]} step={extra[step]} | {message}"
)
# 独立的 Loguru 示例；项目中使用该模板时，修改日志模块中的格式常量。
setup_logging(LoggerConfig(console_enabled=False, file_enabled=False))
logger.add(sys.stderr, format=training_format)
with logger.contextualize(epoch="-", step="-"):
    with logger.contextualize(epoch=1, step=20):
        logger.info("loss={:.4f}", 0.1234)
logger.complete()
```

依赖安装：

```bash
uv sync
# 在其他项目中单独安装：
uv add loguru
```

训练中建议记录摘要，避免整批打印图像张量：

```python
logger.info("epoch={}，step={}，loss={:.4f}", 1, 20, 0.123456)
logger.info("images={}，dtype={}", (2, 3, 384, 640), "torch.float32")
logger.bind(experiment="dinov3_detr", epoch=1).debug("当前批次检查完成")
```

## 1. 导入即可使用

Loguru 提供共享的全局对象，无需自己创建 `Logger` 或调用 `getLogger()`。默认已添加一个输出到 `sys.stderr` 的处理器，其默认最低级别为 DEBUG；项目的配置函数将最低级别设为 INFO。

```python
logger.info("数据加载完成")
logger.debug("图片编号：{}", "road_001")
```

多处 `from loguru import logger` 会使用同一套核心配置；`bind()` 等方法返回的带上下文对象也共享输出目标。

## 2. add()：统一配置输出目标、格式和过滤

`sink` 指日志最终交给谁处理；`add()` 注册一个处理器，返回整数 ID。可以同时存在多个处理器，各自有独立的级别、格式和过滤规则。

```python
logger.remove()
console_id = logger.add(sys.stderr, level="INFO", format="{level} | {message}")
file_id = logger.add("logs/debug.log", level="DEBUG", encoding="utf-8")
logger.debug("仅写入文件")
logger.info("终端和文件都能看到")
logger.remove(file_id)  # 仅停止文件输出
```

### 输出目标支持哪些类型

| 类型 | 例子 | 说明 |
|---|---|---|
| 文件路径 | `"logs/train.log"`、`Path(...)` | 支持文件轮转等附加参数 |
| 文件类对象 | `sys.stderr`、`StringIO()` | 至少提供 `write()` |
| 普通函数 | `save_message` | 接收格式化后的 message |
| 协程函数 | `async def send(...)` | 使用运行中的事件循环 |
| 标准库处理器 | `logging.StreamHandler(...)` | Loguru 转换为 LogRecord 后交给它 |

自定义函数收到的 message 可当作字符串，同时可读取 `message.record`：

```python
records = []
def collect(message):
    records.append(message.record)

logger.remove()
handler_id = logger.add(collect)
logger.info("批次 {} 已处理", 3)
assert records[0]["message"] == "批次 3 已处理"
logger.remove(handler_id)
```

### 过滤日志

```python
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    filter=lambda record: record["extra"].get("stage") == "train",
)
logger.bind(stage="train").info("显示这条")
logger.bind(stage="val").info("过滤这条")
```

`filter` 也可以是模块名前缀字符串，或“模块名 → 最低级别 / 布尔值”的字典：

```python
logger.remove()
logger.add(sys.stderr, level=0, filter={"": "INFO", "src.datasets": "DEBUG", "third_party": False})
```

字典会按最近的父模块匹配。`level` 仍是处理器的总门槛，所以本例使用 `level=0`，避免提前挡住 DEBUG。仅把门槛设为 INFO 会接收 INFO 及以上级别，并非“只接收 INFO”。

### 动态格式

```python
def formatter(record):
    if record["level"].no >= 40:
        return "{time:HH:mm:ss} | {level} | {message}\n{exception}"
    return "{message}\n{exception}"

logger.remove()
logger.add(sys.stderr, format=formatter)
logger.info("简洁输出")
```

字符串格式由 Loguru 自动补换行和异常字段；函数格式需要自己包含 `\n` 与 `{exception}`。格式函数应返回模板，不要把原始消息拼进模板再格式化。

## 3. 文件轮转、保留和压缩

```python
logger.remove()
handler_id = logger.add(
    "logs/train_{time:YYYY-MM-DD_HH-mm-ss}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    diagnose=False,
)
logger.info("开始本次实验")
logger.remove(handler_id)
```

| 参数 | 常用写法 | 行为 |
|---|---|---|
| `rotation` | `"10 MB"`、`"1 day"`、`"00:00"` | 满足条件时关闭当前文件，切换新文件 |
| `retention` | `"7 days"`、`5` | 按最大年龄或保留文件数清理历史日志 |
| `compression` | `"zip"`、`"gz"`、`"tar.gz"` | 归档已结束的日志文件 |

轮转条件在写入日志时检查，不是独立的定时任务。保留与压缩在轮转时执行；未设置轮转时，也可在处理器关闭时执行。`retention=5` 表示保留文件数，不是五天。

可传入自定义回调：

- `rotation(message, file) -> bool`：决定当前是否轮转。
- `retention(paths) -> None`：处理匹配到的历史文件列表。
- `compression(path) -> None`：处理需要归档的单个文件。

这些回调及 sink 内不要再次向同一个日志链路记录消息，以免发生递归或死锁。支持的压缩格式包括 `gz`、`bz2`、`xz`、`lzma`、`tar`、`tar.gz`、`tar.bz2`、`tar.xz`、`zip`。

文件的其他设置见文末 `add()` 参数表。

## 4. 使用花括号格式化

```python
logger.info("样本 {}，目标数 {}", "road_001", 3)
logger.info("epoch={epoch}，loss={loss:.4f}", epoch=2, loss=0.123456)
logger.info("字典内容：{}", {"width": 1920, "height": 1080})
```

使用 `{}` 或命名字段，与 `str.format()` 的写法一致。不要沿用标准 logging 的 `%s` 占位符。

关键字参数默认还会加入日志的 `extra`，可用 `logger.opt(capture=False)` 关闭这一行为；这不会阻止它们参与消息格式化。把外部内容作为参数传入，避免把包含花括号的外部内容直接当作格式模板。

## 5. 捕获函数和线程入口的异常

`catch()` 可以作装饰器或上下文管理器：

```python
@logger.catch(reraise=True)
def calculate_loss():
    raise ValueError("示例：目标框格式错误")

try:
    calculate_loss()
except ValueError:
    pass  # 演示结束；实际入口通常让异常继续传播
```

```python
try:
    with logger.catch(reraise=True):
        raise RuntimeError("示例：训练失败")
except RuntimeError:
    pass
```

线程内部的异常需要在线程入口函数内使用 `catch()`；仅在创建线程的外层使用它，不能自动捕获另一个线程执行时抛出的异常。

| 参数 | 用途 |
|---|---|
| `exception` | 要捕获的异常类型或类型元组，默认 `Exception` |
| `level` | 捕获后按什么级别记录，默认 ERROR |
| `reraise` | 记录后是否继续抛出，默认 False |
| `onerror` | 记录后调用回调，参数为异常对象 |
| `exclude` | 指定不处理、直接继续传播的异常类型 |
| `default` | 装饰器捕获且不重抛时，函数返回的默认值 |
| `message` | 自定义异常提示，可引用 `{record[...]}` |

命令行程序通常使用 `reraise=True` 保留失败状态。`catch()` 默认可能吞掉异常，不能把“已记录”误认为“执行成功”。`logger.add(..., catch=True)` 则控制 sink 自身出错时的行为，与这里的业务异常捕获不同。

## 6. 彩色终端输出

```python
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)
logger.opt(colors=True).info("<cyan>训练阶段</cyan>：开始")
```

`<red>`、`<green>`、`<cyan>`、`<bold>` 等标签用于格式；`<level>` 会按日志级别选择颜色，`</>` 可以关闭最近的标签。`colorize=None` 自动判断终端，True 强制启用，False 去掉颜色。

消息正文中的颜色标签要用 `opt(colors=True)` 解析。格式化参数中的内容不会作为颜色标签解析，因此传入的文件名或外部字符串可以与正文标签分开处理。

## 7. 线程、进程和异步协作

### 7.1 线程与后台队列

默认处理器使用锁协调同一进程中的多线程写入。开启 `enqueue=True` 后，调用线程把消息交给进程安全队列，由后台消费者处理；它不会使业务代码本身变成异步。

```python
logger.remove()
handler_id = logger.add("logs/queued.log", enqueue=True)
logger.info("交给后台处理")
logger.complete()  # 等待已入队消息处理完毕
logger.remove(handler_id)
```

同步代码中的 `logger.complete()` 等待队列排空；有协程 sink 时，还需等待它返回的可等待对象。

### 7.2 多进程，特别是 macOS 的 spawn

父进程配置好共享的队列处理器，将配置后的 logger 传给子进程，队列使用的上下文要与创建进程时一致。下面的完整示例应保存为 `.py` 文件执行：

```python
import multiprocessing as mp
from loguru import logger

def worker(shared_logger):
    shared_logger.info("子进程处理完成")
    shared_logger.complete()

if __name__ == "__main__":
    context = mp.get_context("spawn")
    logger.remove()
    handler_id = logger.add("logs/workers.log", enqueue=True, context=context)
    process = context.Process(target=worker, args=(logger,))
    process.start()
    process.join()
    logger.complete()
    logger.remove(handler_id)
    assert process.exitcode == 0
```

`enqueue=True` 不会让所有独立启动的解释器自动共享同一队列。避免各 worker 分别创建带轮转功能的同名文件处理器；进程启动与 logger 配置应置于 `if __name__ == "__main__":` 下。使用 `multiprocessing.Pool` 时可通过初始化函数将共享 logger 传给 worker。

### 7.3 协程输出目标

```python
import asyncio

received = []
async def async_sink(message):
    await asyncio.sleep(0)
    received.append(str(message))

async def main():
    logger.remove()
    handler_id = logger.add(async_sink)
    logger.info("异步写入示例")
    await logger.complete()
    logger.remove(handler_id)
    assert len(received) == 1

asyncio.run(main())
```

协程 sink 需要活动的事件循环。`loop=None` 时使用记录日志时运行中的循环；没有运行中的循环，任务可能被丢弃。退出循环前执行 `await logger.complete()`，确保当前循环中的日志任务结束；该方法不会替你等待其他事件循环的任务。

## 8. 详细异常堆栈

```python
logger.remove()
logger.add(sys.stderr, backtrace=True, diagnose=False)
try:
    int("not_a_number")
except ValueError:
    logger.exception("数据转换失败")
```

- `logger.exception()` 在 `except` 中使用，以 ERROR 级别记录当前异常。
- `logger.opt(exception=True).warning(...)` 可以用其他级别记录当前异常。
- `backtrace=True` 补充捕获点以上的调用链，帮助定位错误来源。
- `diagnose=True` 会尝试显示栈帧中变量的值；实际部署建议 False，避免输出张量巨量内容或凭据。项目配置使用 False。

```python
try:
    1 / 0
except ZeroDivisionError:
    logger.opt(exception=True).warning("本次计算失败，准备重试")
```

`logger.error("失败")` 本身只输出消息，不会自动附带当前异常，也不会抛出异常或终止训练。

## 9. 结构化日志与上下文

### 9.1 JSON 输出

```python
import json

messages = []
logger.remove()
logger.add(messages.append, serialize=True)
logger.bind(epoch=2).info("批次完成")
payload = json.loads(messages[0])
assert payload["record"]["extra"]["epoch"] == 2
```

`serialize=True` 输出 JSON 字符串，包含可读的 `text` 和结构化 `record`，适合后续分析。它不会把字符串消息自动拆成业务字段，需要查询的字段建议放到 `extra`。

### 9.2 bind()：固定在某个 logger 对象上的字段

```python
logger.remove()
logger.add(sys.stderr, format="{extra[experiment]} | {extra[epoch]} | {message}")
training_log = logger.bind(experiment="dinov3_detr", epoch=1)
training_log.info("开始训练")
training_log.bind(epoch=2).info("下一轮")
```

`bind()` 返回新对象，不改变原对象的绑定字段，也不增加新的输出处理器。格式引用 `extra` 字段时，要确保每条日志都提供该字段，或通过 `configure(extra=...)` 设置默认值。

### 9.3 contextualize()：当前上下文中的临时字段

```python
logger.remove()
logger.configure(extra={"stage": "idle"})
logger.add(sys.stderr, format="{extra[stage]} | {message}")
with logger.contextualize(stage="validation"):
    logger.info("进入验证")
logger.info("退出后恢复 idle")
```

上下文使用 `contextvars` 隔离线程及异步任务的状态，并在退出时恢复；它不会自动将字段传播到新进程。

### 9.4 patch()：记录日志时动态修改 record

```python
import os

def attach_pid(record):
    record["extra"]["worker_pid"] = os.getpid()

logger.remove()
logger.add(sys.stderr, format="{extra[worker_pid]} | {message}")
worker_log = logger.patch(attach_pid)
worker_log.info("处理图片")
```

回调原地修改 record，通常放入 `extra`。`patch()` 返回的对象保留同一套输出配置，便于按用途附加字段；全局回调可以通过 `configure(patcher=...)` 配置。

## 10. 延迟计算与 opt() 的所有选项

普通函数参数会在调用 logger 之前计算，隐藏 DEBUG 并不阻止这些计算。耗时的调试统计可以延迟执行：

```python
logger.remove()
logger.add(sys.stderr, level="INFO")

def expensive_summary():
    raise AssertionError("DEBUG 被过滤时不应执行这里")

logger.opt(lazy=True).debug("统计结果：{}", expensive_summary)
```

`lazy=True` 下，需要格式化的参数应传入无参函数或 lambda。它在日志级别达到处理器门槛时计算，不保证后续自定义 filter 一定接受该消息。

| `opt()` 选项 | 用途 |
|---|---|
| `exception` | 附带当前异常、指定异常对象或异常三元组 |
| `record=True` | 允许正文使用 `{record[line]}` 等字段 |
| `lazy=True` | 延迟调用参数函数 |
| `colors=True` | 解析正文中的颜色标签 |
| `raw=True` | 跳过输出格式，原样发送消息，需要自己处理换行 |
| `capture=False` | 关键字参数只用于格式化，不加入 extra |
| `depth=1` | 使用上一层调用位置，适用于自定义日志包装函数 |
| `ansi` | 已弃用，改用 `colors` |

```python
logger.opt(record=True).info("当前位置：{record[name]}:{record[line]}")
logger.opt(raw=True).info("不带前缀的一行\n")
logger.opt(capture=False).info("epoch={epoch}", epoch=1)

def log_step(message):
    logger.opt(depth=1).info(message)

log_step("显示 log_step 的调用位置")
```

同一次调用的选项放在一个 `opt()` 中。连续 `.opt(...).opt(...)` 会重置前一次选项，不能按累加理解。

## 11. 完整内置级别与自定义级别

| 级别 | 数值 | 方法 | 适合记录 |
|---|---|---|---|
| TRACE | 5 | `logger.trace()` | 非常细的执行步骤 |
| DEBUG | 10 | `logger.debug()` | 样本编号、内部状态 |
| INFO | 20 | `logger.info()` | 实验配置、正常进度 |
| SUCCESS | 25 | `logger.success()` | 检查通过、保存完成 |
| WARNING | 30 | `logger.warning()` | 缺少标注、可恢复问题 |
| ERROR | 40 | `logger.error()` | 当前任务失败 |
| CRITICAL | 50 | `logger.critical()` | 严重故障 |

```python
logger.level("METRIC", no=21, color="<cyan>", icon="M")
logger.log("METRIC", "验证 mAP={:.3f}", 0.512)
logger.log(20, "直接按数值记录 INFO 严重程度的日志")
info_level = logger.level("INFO")
assert info_level.no == 20
```

数值调用 `log()` 不等于注册了一个新级别名称。已有级别可以更新颜色或图标，不能修改其数值。最低输出级别属于处理器，不存在 `logger.setLevel()`；要更改已有处理器的门槛，移除后重新添加，或使用动态过滤函数。

## 12. 时间格式和时区

```python
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS ZZ} | {message}")
logger.info("显示本地时间和时区偏移")
```

| 标记 | 含义 |
|---|---|
| `YYYY`、`YY` | 四位、两位年份 |
| `MM`、`DD` | 月、日 |
| `HH`、`mm`、`ss` | 时、分、秒，注意大小写 |
| `SSS`、`SSSSSS` | 毫秒、微秒位数 |
| `Z`、`ZZ` | 带冒号、不带冒号的时区偏移 |
| `[文字]` | 将时间格式中的内容当作字面量 |
| `!UTC` | 在时间格式结尾要求转为 UTC |

```python
logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD[T]HH:mm:ss.SSS!UTC} | {message}")
logger.info("记录 UTC 时间")
```

`{time}` 来自日志产生时的带时区 datetime；文件名中的 `{time}` 用于创建该日志文件的时间，不代表每写一条消息都新建一个文件。更完整的时间标记查阅 API 的时间格式说明。

## 13. 应用、脚本和库的配置方式

### 批量配置

```python
handler_ids = logger.configure(
    handlers=[{"sink": sys.stderr, "level": "INFO", "format": "{extra[experiment]} | {message}"}],
    levels=[{"name": "CHECKPOINT", "no": 26, "color": "<green>", "icon": "C"}],
    extra={"experiment": "default"},
    activation=[("", True), ("third_party.noisy", False)],
)
logger.log("CHECKPOINT", "权重已保存")
```

`configure()` 支持 `handlers`、`levels`、`extra`、`patcher`、`activation`。指定 `handlers` 会替换原有处理器；指定 `extra` 会替换核心默认字段。`patcher` 是全局 record 修改函数，先于 `patch()` 的回调执行。`activation` 按顺序启用或禁用模块。

### 库按模块启用和禁用

```python
logger.disable("src.datasets")
# 该模块及其子模块产生的 Loguru 日志会被禁用。
logger.enable("src.datasets")
```

应用在入口配置输出。可复用库通常只使用 logger，并默认禁用自己包名的日志，避免擅自改变调用方的处理器；应用可以再 `enable()`。`disable()` 按日志 record 的模块名匹配，与文件路径字符串不同。

Loguru 的 `configure()` 接收 Python 字典；直接从 YAML/JSON 文件读取并构造日志配置，可使用普通文件加载代码，或按需安装第三方 `loguru-config`。项目现有 Pydantic 配置不会自动变成 Loguru 配置。

## 14. 与标准库 logging 互通

### Loguru 交给标准 Handler

```python
import logging

logger.remove()
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("标准处理器：%(message)s"))
logger.add(handler, format="{message}")
logger.info("仍由 Loguru 产生，交给标准 Handler 输出")
```

可以替换成其他 `logging.Handler`，包括网络、系统日志或自定义处理器。需要转交标准 logging 的 logger 层级时，可用下面的桥接：

```python
import logging

class ForwardToStandard(logging.Handler):
    def emit(self, record):
        target = logging.getLogger(record.name)
        target.handle(record)

logger.remove()
logging.basicConfig(level=logging.INFO, force=True)
logger.add(ForwardToStandard(), format="{message}")
logger.info("交给标准 logging 的处理链")
```

### 标准 logging 交给 Loguru

```python
import logging

class ForwardToLoguru(logging.Handler):
    def emit(self, record):
        try:
            target_level = logger.level(record.levelname).name
        except ValueError:
            target_level = record.levelno
        logger.opt(exception=record.exc_info).log(target_level, record.getMessage())

logger.remove()
logger.add(sys.stderr, format="{level} | {message}")
logging.basicConfig(handlers=[ForwardToLoguru()], level=logging.INFO, force=True)
logging.getLogger("third_party").warning("第三方库的一条警告")
```

这个简化适配器保留消息、级别及异常；Loguru 的代码位置显示在适配器内部。需要保留原始调用位置时，可按官方配方计算调用深度并使用 `opt(depth=...)`。

两个方向的桥接不要同时串成闭环。`basicConfig(force=True)` 会替换标准 root logger 的现有处理器，应只在应用入口按需调用。第三方库如果关闭了传播或使用独立处理器，还需单独配置该库。

## 15. 用环境变量定制默认行为

在导入 Loguru 之前设置，例如在终端执行：

```bash
LOGURU_LEVEL=INFO LOGURU_FORMAT='{time:HH:mm:ss} | {level} | {message}' uv run python -c 'from loguru import logger; logger.info("环境变量配置生效")'
```

| 环境变量 | 用途 |
|---|---|
| `LOGURU_LEVEL` | 默认输出最低级别 |
| `LOGURU_FORMAT` | 默认格式 |
| `LOGURU_COLORIZE` | 默认颜色行为 |
| `LOGURU_DIAGNOSE` | 是否展示异常中的变量值，例如 `NO` |
| `LOGURU_BACKTRACE` | 是否扩展异常堆栈 |
| `LOGURU_ENQUEUE` | 默认是否启用队列 |
| `LOGURU_AUTOINIT` | 是否自动添加默认处理器，例如 `False` |
| `LOGURU_DEBUG_COLOR` | DEBUG 级别的默认颜色 |
| `LOGURU_TRACE_ICON` | TRACE 级别的默认图标 |

一般形式为 `LOGURU_<参数名>` 和 `LOGURU_<级别>_<属性>`。这些变量影响默认值，不会覆盖代码中显式传入的参数。项目通过 `setup_logging(config.logger)` 显式指定级别，因此应修改 YAML 中的 `logger.level`，只修改 `LOGURU_LEVEL` 不会覆盖这一参数。

## 16. parse()：解析已经写出的文本日志

```python
from io import StringIO

text_log = StringIO("20|batch=3 loss=0.125\n")
pattern = r"(?P<level>\d+)\|batch=(?P<batch>\d+) loss=(?P<loss>[\d.]+)"
entries = list(logger.parse(
    text_log,
    pattern,
    cast={"level": int, "batch": int, "loss": float},
    chunk=65536,
))
assert entries == [{"level": 20, "batch": 3, "loss": 0.125}]
```

可传入路径或已打开的文件对象；正则中的命名组成为结果字典的键。`cast` 可以是字段到转换函数的字典，也可以是原地修改结果字典的函数。它返回迭代器，适合分块读取已有文件。

`parse()` 不会自动识别任意日志格式。若文件由 `serialize=True` 产生，通常逐行 `json.loads()` 更直接。

## 17. 对接通知服务

Loguru 可以把 ERROR 及以上日志交给通知函数，官方示例使用第三方 Apprise；发送邮件或消息的能力由通知库和服务提供，不是 Loguru 自带邮箱客户端。

先按需安装：

```bash
uv add apprise
```

下面是需要自行配置服务地址的集成示例，不属于项目默认启用的输出：

```python
import os
import apprise

notifier = apprise.Apprise()
notifier.add(os.environ["ALERT_SERVICE_URL"])

def notify_error(message):
    notifier.notify(title="检测实验异常", body=str(message))

handler_id = logger.add(
    notify_error,
    level="ERROR",
    filter={"apprise": False},
)
# 之后产生的 ERROR/CRITICAL 才会调用通知函数。
```

只在确认需要发送通知、配置好自己的服务后运行。过滤 Apprise 自身日志可减少递归调用；真实项目还应考虑重复告警、发送失败和通知频率。本次文档验证不实际发送通知。

## 18. 公开方法完整索引

以下覆盖本地 0.7.3 对象上的全部 24 个公开可调用方法，包括两个弃用别名。

| 方法 | 使用位置或用途 |
|---|---|
| `logger.add()` | 注册终端、文件、函数、协程或标准 Handler，第 2、3、7 节 |
| `logger.remove()` | 按处理器 ID 移除；不传 ID 时移除全部，第 2 节 |
| `logger.complete()` | 等待队列和协程任务，第 7 节 |
| `logger.trace()` | TRACE 日志，第 11 节 |
| `logger.debug()` | DEBUG 日志，第 11 节 |
| `logger.info()` | INFO 日志，第 1、4 节 |
| `logger.success()` | SUCCESS 日志，第 11 节 |
| `logger.warning()` | WARNING 日志，第 11 节 |
| `logger.error()` | ERROR 日志，第 8、11 节 |
| `logger.critical()` | CRITICAL 日志，第 11 节 |
| `logger.log()` | 按名称或数值选择级别，第 11 节 |
| `logger.exception()` | ERROR 级别记录当前异常，第 8 节 |
| `logger.catch()` | 异常装饰器或上下文管理器，第 5 节 |
| `logger.opt()` | 控制单次日志的生成选项，第 10 节 |
| `logger.bind()` | 返回绑定额外字段的对象，第 9 节 |
| `logger.contextualize()` | 临时设置当前上下文字段，第 9 节 |
| `logger.patch()` | 返回使用 record 回调的对象，第 9 节 |
| `logger.level()` | 查询、增加或调整级别样式，第 11 节 |
| `logger.disable()` | 禁用模块及其子模块日志，第 13 节 |
| `logger.enable()` | 重新启用模块日志，第 13 节 |
| `logger.configure()` | 批量配置处理器、级别与默认字段，第 13 节 |
| `logger.parse()` | 按正则解析文本日志，第 16 节 |
| `logger.start()` | 已弃用别名，使用 `logger.add()` |
| `logger.stop()` | 已弃用别名，使用 `logger.remove()` |

`exception()` 不单独增加一个级别。不要直接实例化 `Logger`，也不要依赖 `_core` 等私有属性。`start()`、`stop()` 仅为说明历史兼容接口，新代码不使用。

## 19. add() 参数速查

### 通用参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `sink` | 必填 | 输出目标 |
| `level` | `"DEBUG"` | 该处理器最低级别 |
| `format` | 时间、级别、位置和消息的彩色模板 | 字符串模板或返回模板的函数 |
| `filter` | `None` | 函数、模块名前缀或模块级别字典 |
| `colorize` | `None` | 自动判断是否转换颜色标签 |
| `serialize` | `False` | 输出 JSON |
| `backtrace` | `True` | 扩展异常调用链 |
| `diagnose` | `True` | 在异常中显示变量值；项目显式设为 False |
| `enqueue` | `False` | 用进程安全队列分发日志 |
| `context` | `None` | 队列使用的 multiprocessing 上下文对象或名称 |
| `catch` | `True` | 是否拦截 sink 内部处理日志时的异常 |
| `**kwargs` | 无 | 文件或协程专用参数 |

默认值可受环境变量影响；项目配置函数传入的值以代码为准。使用 `catch=False` 时，输出目标自身的错误会继续抛给调用方；这不改变普通业务异常的捕获方式。

### 文件路径专用参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `rotation` | `None` | 按大小、时刻、周期或回调轮转 |
| `retention` | `None` | 按文件数量、年龄或回调保留 |
| `compression` | `None` | 压缩格式或自定义归档回调 |
| `delay` | `False` | 首条日志写入时再创建文件 |
| `watch` | `False` | 检查文件是否被外部删除或替换，必要时重新打开 |
| `mode` | `"a"` | 文件打开模式，默认追加 |
| `buffering` | `1` | 缓冲策略，默认按行缓冲 |
| `encoding` | `"utf8"` | 文本编码 |
| 其他 `open()` 参数 | 按 open 默认值 | 例如 `errors`、`opener` |

### 协程专用参数

| 参数 | 默认值 | 用途 |
|---|---|---|
| `loop` | `None` | 指定异步输出使用的事件循环 |

## 20. record 字段速查

格式模板、filter、patcher 和 sink 的 `message.record` 可使用这些字段：

| 字段 | 内容与示例 |
|---|---|
| `time` | 产生消息的时间，`{time:HH:mm:ss}` |
| `elapsed` | 从 Loguru 启动起经过的时间 |
| `level` | 级别对象，`{level.name}`、`{level.no}`、`{level.icon}` |
| `message` | 已完成参数格式化的消息正文 |
| `name` | 日志来源的模块名 |
| `module` | 来源文件去掉扩展名后的名字 |
| `file` | 来源文件信息，`{file.name}`、`{file.path}` |
| `function` | 来源函数名 |
| `line` | 来源行号 |
| `process` | 进程信息，`{process.id}`、`{process.name}` |
| `thread` | 线程信息，`{thread.id}`、`{thread.name}` |
| `exception` | 异常信息或 None，用于格式化堆栈 |
| `extra` | 自定义字段，`{extra[epoch]}` |

`format` 中可直接使用上述字段；消息正文要访问整条 record 时，使用 `opt(record=True)`。

## 21. 项目使用建议与常见问题

| 情况 | 处理方式 |
|---|---|
| 终端重复打印 | 检查是否反复 add；入口配置一次，替换默认输出时先 remove |
| INFO 显示、DEBUG 不显示 | 将 `logger.level` 改为 DEBUG，或使用 `setup_logging(LoggerConfig(level="DEBUG"))` |
| 只想查看图片编号 | DEBUG 记录 ID，INFO 记录批次形状和目标数量 |
| 日志与进度条相互覆盖 | 继续使用项目的 `tqdm.write()` sink |
| 打印 ERROR 后程序还继续运行 | 日志级别不会改变流程，需要显式 raise 或正确返回 |
| 退出时后台日志未写完 | 同步队列调用 complete；协程 sink 使用 await complete |
| extra 字段缺失导致格式报错 | 用 configure 设置默认值，或确保每条日志都绑定字段 |
| 多 worker 重复配置输出 | 入口加 main 判断，并明确选择共享队列还是各进程独立输出 |
| 希望收集第三方库日志 | 使用 logging 适配器，Loguru 不会默认接管它们 |
| 需要重定向 print 或 warnings | 按官方配方配置流适配器；不要把日志再输出到同一个被重定向的流 |
| 移动了日志配置文件 | 同步修改导入路径，当前为 `src.utils.logger` |

高级配方还包括标准输出与 warnings 捕获、独立文件分流、自定义异常显示和自定义序列化等；它们由以上 API 组合实现，详见官方 Recipes。

## 22. 官方来源

- [Overview：官方特性与入门](https://loguru.readthedocs.io/en/stable/overview.html)
- [Logger API：公开接口、参数与格式规则](https://loguru.readthedocs.io/en/stable/api/logger.html)
- [Recipes：并发、适配器与高级组合示例](https://loguru.readthedocs.io/en/stable/resources/recipes.html)
- [Troubleshooting：级别、重复输出与格式问题](https://loguru.readthedocs.io/en/stable/resources/troubleshooting.html)

以上为官方 stable 文档入口；实际项目按锁文件中的 0.7.3 使用。升级 Loguru 后，应重新核对公开接口和并发相关说明。
