"""训练入口：python train.py --help。"""

from src.trainer.cli import build_parser, config_overrides


def main(argv=None):
    mode = "train"
    args = build_parser(mode=mode).parse_args(argv)
    from src.trainer.engine import Engine

    Engine(config_overrides(args, mode=mode)).train()


if __name__ == "__main__":
    main()
