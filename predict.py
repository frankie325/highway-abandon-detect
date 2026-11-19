"""本地图片预测入口。"""

from src.trainer.cli import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(mode="predict")
    args = parser.parse_args(argv)

    from src.trainer.engine import Engine

    try:
        Engine(outputs=args.outputs).predict(
            args.images, batch_size=args.batch_size, score_threshold=args.score_threshold
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
