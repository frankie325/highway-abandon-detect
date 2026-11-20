"""独立验证入口：python eval.py --help。"""

from src.trainer.cli import build_parser, config_overrides


def main(argv=None):
    mode = "eval"
    args = build_parser(mode=mode).parse_args(argv)
    from src.trainer.engine import Engine

    Engine(config_overrides(args, mode=mode)).eval()


if __name__ == "__main__":
    main()
