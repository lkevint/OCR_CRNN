Authors: Kevin Liu, Derek Gubbens, Jiajian Huang

This project is meant to explore the applicability of using CRNN architecture pretrained on a well labeled synthetic dataset that is heavily altered to imitate possible real world shapes and images. Our choice of dataset is [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/). Text examples can be extremely inconsistent in size, shape, and font.

**Results:**
We trained our model on a subset of $2^{21}$ training images for 5 epochs and achieved a training accuracy of 76% and a testing accuracy of 70%. We believe this performance is very satisfactory, as the accuracy is based on the rate of exact character matching, and many of the images in the dataset are very ambiguous. Below are some results of inference on testing images.


**Dataset:**
[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)

# CRNN OCR

This project contains:
- `train_crnn.py` for training
- `infer_crnn.py` for running inference on a folder of images

## Setup

Install the requirements:

```bash
pip install -r requirements.txt
```
If you intend to use CUDA, change the PyTorch arguments appropriately.

## Training

Example usage:

```bash
python train_crnn.py --epochs 5 --samples 8192 --batch_size 32 --root path_to/90kDICT32px 
```

Arguments:
- `--epochs`: number of training epochs (default: 5)
- `--samples`: number of training samples to use (default: 8192)
- `--batch_size`: batch size (default: 32)
- `--root`: path to the dataset root

Notes:
- `root` is assumed to be the `90kDICT32px` folder found inside MJSynth download
- `annotation_train.txt` and `annotation_test.txt` are assumed to be inside `root`

## Inference

Example usage:

```bash
python infer_crnn.py --model models/CRNN_0.pth --image_dir my_images
```

Arguments:
- `--model`: path to a saved model file
- `--image_dir`: folder containing images to read

The script prints the predicted text for each supported image in the folder.