from torchvision.models.squeezenet import Fire
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import numpy as np


class Wasserstein_SqueezeNet(nn.Module):
    '''
    An object representing the neural network described in "Automatic Color Correction for Multisource Remote Sensing Images with Wasserstein CNN" (J. Guo et al).

    This is a modified version of SqueezeNet version 1.1 (see https://paperswithcode.com/model/squeezenet?variant=squeezenet-1-1 for details).

    The purpose of this network is to predict reference histograms suitable for histogram matching with given source images.
    '''

    def __init__(self, dropout_probability=0.5):
        '''
        Initializes a SqueezeNet (1.1) implementation designed to predict suitable reference histograms.
        '''

        super(Wasserstein_SqueezeNet, self).__init__()

        self.eighth_pool = nn.AvgPool2d(kernel_size=15, stride=8)

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2),
            nn.ReLU(inplace=True)
        )

        self.quarter_pool = nn.AvgPool2d(kernel_size=7, stride=4)

        self.initial_fire_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64)
        )

        self.half_pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.interior_fire_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128)
        )

        # No pooling necessary between these blocks.

        last_conv = nn.Conv2d(512, 512, kernel_size=1)
        self.final_fire_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
            nn.Dropout(p=dropout_probability),
            last_conv,
            nn.ReLU(inplace=True)
        )

        self.deconvolve = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2)

        self.outputs = nn.ModuleList([nn.Sequential(
            nn.Linear(27 * 27 * (3 + 96 + 128 + 256 + 512), 256), nn.Softmax(dim=1)) for idx in range(3)])

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is last_conv:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        '''
        Predicts suitable reference histograms for a tensor of source images.
        '''
        # assert x.shape[1:] == (self.channels, *self.source_tensor_dimensions)
        convolved = self.conv_block(x)
        initial_fired = self.initial_fire_block(convolved)
        interior_fired = self.interior_fire_block(initial_fired)
        final_fired = self.final_fire_block(interior_fired)
        # print(self.eighth_pool(x).shape)  # torch.Size([40, 3, 27, 27])
        # print(self.quarter_pool(convolved).shape)  # torch.Size([40, 96, 27, 27])
        # print(self.half_pool(initial_fired).shape)  # torch.Size([40, 128, 27, 27])
        # print(interior_fired.shape)  # torch.Size([40, 256, 27, 27])
        # print(self.deconvolve(final_fired).shape)  # torch.Size([40, 512, 27, 27])
        # print("**********************")
        # print(torch.flatten(self.eighth_pool(x), start_dim=1).shape)  # torch.Size([40, 2187])
        # print(torch.flatten(self.quarter_pool(convolved), start_dim=1).shape)  # torch.Size([40, 69984])
        # print(torch.flatten(self.half_pool(initial_fired), start_dim=1).shape)  # torch.Size([40, 93312])
        # print(torch.flatten(interior_fired, start_dim=1).shape)  # torch.Size([40, 186624])
        # print(torch.flatten(self.deconvolve(final_fired), start_dim=1).shape)  # torch.Size([40, 373248])
        concatenated = torch.cat(
            [torch.flatten(self.eighth_pool(x), start_dim=1), torch.flatten(self.quarter_pool(convolved), start_dim=1),
             torch.flatten(self.half_pool(initial_fired), start_dim=1), torch.flatten(interior_fired, start_dim=1),
             torch.flatten(self.deconvolve(final_fired), start_dim=1)], dim=1)  # torch.Size([40, 725355])
        out = torch.stack([output(concatenated) for output in self.outputs], dim=1)  # torch.Size([40, 3, 256])
        return out


if __name__ == '__main__':
    model = Wasserstein_SqueezeNet()

    image1 = Image.open(r"D:\Datasets\yunse\train\crop_img_1_34.tif")
    image2 = Image.open(r"D:\Datasets\yunse\train\crop_img_31_12.tif")
    image3 = Image.open(r"D:\Datasets\yunse\train\crop_img_10_33.tif")
    image4 = Image.open(r"D:\Datasets\yunse\train\crop_img_10_32.tif")
    image_a = np.concatenate(
        (np.array(image1.resize((224, 224))), np.array(image2.resize((224, 224))), np.array(image3.resize((224, 224))),
         np.array(image4.resize((224, 224))),),
        axis=0)
    image = Image.fromarray(np.array(image_a))
    toTensor = transforms.ToTensor()  # 实例化一个toTensor
    image_tensor = toTensor(image)
    image_tensor = image_tensor.reshape(4, 3, 224, 224)
    output1 = model(image_tensor)
    # print(output1, output2, output3)
    print("out", output1.shape)

    # print(model)
