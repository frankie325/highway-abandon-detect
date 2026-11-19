"""训练入口：python train.py --help。"""

from src.trainer.cli import build_parser, config_overrides


def main(argv=None):
    args = build_parser(mode="train").parse_args(argv)
    from src.trainer.engine import Engine

    Engine(config_overrides(args), outputs=args.outputs).train()


if __name__ == "__main__":
    main()
