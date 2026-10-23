from src import DinoV3
from src.utils.logger import setup_logging
from src.config import load_config
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor
import requests
from loguru import logger


def main():
    root = Path(__file__).resolve().parent

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        cache_dir=Path(root) / "cache",
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    # print(inputs)
    model = DinoV3(root=str(root))
    outputs = model(inputs.pixel_values)
    # print(type(outputs))
    # print(outputs.logits)
    # print(outputs[0])
    # print(outputs.keys())
    logger.info("骨干网络输出 {} 层特征图", len(outputs.feature_maps))


if __name__ == "__main__":
    setup_logging(load_config().logger)
    with logger.catch(reraise=True):
        main()
