from src import DinoV2
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor
import requests


def main():
    root = Path(__file__).resolve().parent

    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base", cache_dir=Path(root) / "cache"
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    # print(inputs)
    model = DinoV2(root=str(root))
    outputs = model(inputs.pixel_values)
    # print(type(outputs))
    # print(outputs.logits)
    # print(outputs[0])
    # print(outputs.keys())
    print(len(outputs.feature_maps))


if __name__ == "__main__":
    main()
