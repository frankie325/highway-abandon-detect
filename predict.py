"""本地图片预测入口。"""

from src.trainer.cli import build_parser, config_overrides


def main(argv: list[str] | None = None) -> None:
    mode = "predict"
    parser = build_parser(mode=mode)
    args = parser.parse_args(argv)

    from src.trainer.engine import Engine

    try:
        Engine(config_overrides(args, mode=mode)).predict()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
