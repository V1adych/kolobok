import argparse
import pprint
from torchvision.io import read_image

from tire_vision.thread.pipeline import TireVisionPipeline
from tire_vision.config import TireVisionConfig


def main():
    parser = argparse.ArgumentParser(description="Tire Vision Pipeline")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    config = TireVisionConfig()
    pipeline = TireVisionPipeline(config)

    image = read_image(args.image_path)

    if image.shape[0] == 4:
        image = image[:3, :, :]

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = image

    result = pipeline(image)

    print("Analysis Result:")
    pprint.pprint(result)

if __name__ == "__main__":
    main()
