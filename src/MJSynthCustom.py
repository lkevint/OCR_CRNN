import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from . import process_img, label_utils

# Custom Dataset object using MJSynth for Dataloader
class MJSynthCustom(Dataset):
    def __init__(self, root, annotations_file, samples):
        self.root = root
        self.paths = annotations_file.read_text().split()[::2][0:samples]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.root / self.paths[idx]
        try:
            img = process_img.process_image(Image.open(img_path))
        except OSError as e:
            print(f"Skipping bad image: {img_path} | {e}")
            return self.__getitem__((idx + 1) % len(self))

        label_text = self.paths[idx].split('_')[1]
        label, label_len = label_utils.encode_to_labels(label_text)
        return img, label, torch.tensor(label_len, dtype=torch.long)
