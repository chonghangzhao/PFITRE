import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models


def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0, bias=False
            ),
            nn.GELU(), #nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualDenseBlock_BottleNeck(nn.Module):
    def __init__(self, in_channel=32, inc_channel=32, beta=1.0):
        super().__init__()
        """
        Args:
            num_filters:number of convolution layers
            beta: weight of output added to input, default is 1, so that output:input=1:1 was added for the following layers
        """
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(in_channel + 2*inc_channel, inc_channel, 3, padding=8, dilation=8)
        self.conv4 = nn.Conv2d(in_channel + 3*inc_channel, inc_channel, 3, padding=4, dilation=4)
        self.conv5 = nn.Conv2d(in_channel + 4*inc_channel,  in_channel, 3, padding=2, dilation=2)
        self.conv6 = nn.Conv2d(in_channel + 5*inc_channel,  in_channel, 3, padding=1)
        
        self.gelu = nn.GELU() 
        self.b = beta
        
    def forward(self, x):
        block1 = self.gelu(self.conv1(x))
        block2 = self.gelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.gelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        block4 = self.gelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        block5 = self.gelu(self.conv5(torch.cat((block4, block3, block2, block1, x), dim = 1)))
        out = self.gelu(self.conv6(torch.cat((block5, block4, block3, block2, block1, x), dim = 1)))
        
        return x + self.b * out


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=32, inc_channel=32, beta=0.2):
        super().__init__()
        """
        Args:
            num_filters:number of convolution layers
            beta: weight of output added to input, default is 0.2, so that output:input=0.2:1 was added for the following layers
        """
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel + 2*inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel + 3*inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channel + 4*inc_channel,  in_channel, 3, 1, 1)
        self.lrelu = nn.GELU() 
        self.b = beta
        
    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim = 1))
        
        return x + self.b * out

class PFITRE_net(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = True) -> None:
        """
        Args:
            num_filters:
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool1 = nn.Conv2d(64,64, 2, stride=2)
        self.pool2 = nn.Conv2d(128,128,2, stride=2)

        self.encoder = models.vgg11(pretrained=pretrained).features 
        self.relu = nn.GELU() 
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.ResBlock = ResidualDenseBlock_BottleNeck(num_filters * 8, num_filters * 8)
        self.ResidualDenseBlock = ResidualDenseBlock(num_filters * 8, num_filters * 8)

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 16), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 8), num_filters * 4 * 2, num_filters * 2
        )  
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool1(conv1)))
        conv3s = self.relu(self.conv3s(self.pool2(conv2)))

        ResBlock1 = self.ResBlock(conv3s) 
        ResBlock2 = self.ResidualDenseBlock(ResBlock1)
        ResBlock3 = self.ResBlock(ResBlock2)
        ResBlock4 = self.ResidualDenseBlock(ResBlock3)

        dec3 = self.dec3(torch.cat([ResBlock4, conv3s], 1))    
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) 
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)



