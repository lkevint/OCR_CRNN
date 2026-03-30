import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFile

from src import CRNN, global_variables, label_utils, process_img


ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    crnn_model = CRNN.CRNN(
        input_channels=1,
        output_shape=global_variables.MAX_LABEL_LEN * (len(global_variables.CHAR_LIST) + 1),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    crnn_model.load_state_dict(state_dict)
    crnn_model.eval()
    return crnn_model


def predict_single_image(model, image_tensor):
    with torch.inference_mode():
        pred = model(image_tensor.unsqueeze(0).to(device))
    decoded = label_utils.decode_prediction(pred.cpu())[0]
    return label_utils.decode_to_text(decoded)



def run_custom_directory(model, image_dir):
    image_dir = Path(image_dir)
    image_paths = sorted(
        [
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        ]
    )

    if not image_paths:
        raise ValueError("No supported image files were found in the custom directory.")

    for image_path in image_paths:
        try:
            img_tensor = process_img.process_image(Image.open(image_path))
        except OSError as e:
            print(f"Skipping bad image: {image_path} | {e}")
            continue

        prediction = predict_single_image(model, img_tensor)
        print(f"Image: {image_path.name}")
        print(f"Predicted: {prediction}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    crnn_model = load_model(model_path)
    run_custom_directory(crnn_model, args.image_dir)
