Authors: Kevin Liu, Derek Gubbens, Jiajian Huang

This project is meant to explore the applicability of using CRNN architecture pretrained on a well labeled synthetic dataset that is heavily altered to imitate possible real world shapes and images. Our choice of dataset is [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/). Text examples can be extremely inconsistent in size, shape, and font.

**Dataset:**
[MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)

**Results:**
We trained our model on a subset of $2^{21}$ training images for 5 epochs and achieved a training accuracy of 76% and a testing accuracy of 70%. We believe this performance is very satisfactory, as the accuracy is based on the rate of exact character matching, and many of the images in the dataset are very ambiguous. Below are some results of inference on testing images.

![13_composite_15517](https://github.com/user-attachments/assets/64ba835b-5c81-4cbe-a40f-91ad1ddbc3aa)

Predicted: composite

![176_underpays_82187](https://github.com/user-attachments/assets/440e9cbf-b4df-4339-8369-50732a98ae15)

Predicted: underpays

![173_Ukase_81594](https://github.com/user-attachments/assets/fb1740e7-90d2-4b47-a453-88f8f3a70bb0)

Predicted: Ukase

![9_unabashedly_81675](https://github.com/user-attachments/assets/68577b4b-420f-4c3b-b798-9eb427ddacc3)

Predicted: Uncahelly
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
