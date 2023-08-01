from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        # Encoder
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # Decoder 1
        self.upconv4_dec1 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_dec1 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3_dec1 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_dec1 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2_dec1 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_dec1 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1_dec1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_dec1 = UNet._block(features * 2, features, name="dec1")
        self.conv_dec1 = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # Decoder 2
        self.upconv4_dec2 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4_dec2 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3_dec2 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3_dec2 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2_dec2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2_dec2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1_dec2 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1_dec2 = UNet._block(features * 2, features, name="dec1")
        self.conv_dec2 = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec1_4 = self.upconv4_dec1(bottleneck)
        dec1_4 = torch.cat((dec1_4, enc4), dim=1)
        dec1_4 = self.decoder4_dec1(dec1_4)
        dec1_3 = self.upconv3_dec1(dec1_4)
        dec1_3 = torch.cat((dec1_3, enc3), dim=1)
        dec1_3 = self.decoder3_dec1(dec1_3)
        dec1_2 = self.upconv2_dec1(dec1_3)
        dec1_2 = torch.cat((dec1_2, enc2), dim=1)
        dec1_2 = self.decoder2_dec1(dec1_2)
        dec1_1 = self.upconv1_dec1(dec1_2)
        dec1_1 = torch.cat((dec1_1, enc1), dim=1)
        dec1_1 = self.decoder1_dec1(dec1_1)

        dec2_4 = self.upconv4_dec2(bottleneck)
        dec2_4 = torch.cat((dec2_4, enc4), dim=1)
        dec2_4 = self.decoder4_dec2(dec2_4)
        dec2_3 = self.upconv3_dec2(dec2_4)
        dec2_3 = torch.cat((dec2_3, enc3), dim=1)
        dec2_3 = self.decoder3_dec2(dec2_3)
        dec2_2 = self.upconv2_dec2(dec2_3)
        dec2_2 = torch.cat((dec2_2, enc2), dim=1)
        dec2_2 = self.decoder2_dec2(dec2_2)
        dec2_1 = self.upconv1_dec2(dec2_2)
        dec2_1 = torch.cat((dec2_1, enc1), dim=1)
        dec2_1 = self.decoder1_dec2(dec2_1)

        return torch.sigmoid(self.conv_dec1(dec1_1)), torch.sigmoid(self.conv_dec2(dec2_1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
