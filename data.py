import os
import torch
import numpy as np
from image import image_open, image_normalize
from histo import hist_open, hist_normalize
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SelfDatasetFolder(VisionDataset):
    def __init__(self, imgroot, transform=None):
        super(SelfDatasetFolder, self).__init__(imgroot, transform=transform)
        samples = self.make_dataset(imgroot)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.matches = samples

    def make_dataset(self, imgroot):
        matches = []
        target_imgroot = os.path.join(imgroot, "label")
        for D, R, files in os.walk(target_imgroot):
            for file in files:
                matches.append((os.path.join(imgroot, "shuru", file), os.path.join(D, file)))
        return matches

    def __getitem__(self, index):
        return torch.cat(
            [torch.from_numpy(image_normalize(image_open(self.matches[index][0])).astype(np.float32))]), \
               torch.cat([torch.from_numpy(hist_normalize(hist_open(self.matches[index][1])).astype(np.float32))])

    def __len__(self):
        return len(self.matches)


if __name__ == "__main__":
    imgroot = r'D:\Datasets\1228_same\train'
    dataset = SelfDatasetFolder(imgroot)
    print(dataset)
    print("data num: ", len(dataset))
    sample, path = dataset[0]
    print(sample, path)
    sample, path = dataset[-1]
    print(sample, path)
