import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import warnings
import math
from torch.nn import Softmax


class BCSA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super(BCSA, self).__init__()
        self.avg_pool_channel = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_spatial = nn.AdaptiveAvgPool2d(1)
        self.conv_channel = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.conv_spatial = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y_channel = self.avg_pool_channel(x)
        y_channel = self.conv_channel(y_channel.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_channel = self.sigmoid(y_channel)


        y_spatial = self.avg_pool_spatial(x)
        y_spatial = self.conv_spatial(y_spatial.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_spatial = self.sigmoid(y_spatial)

        y = y_channel * y_spatial

        return x * y.expand_as(x)



class DownBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU()
        )

        self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.downsample = nn.Conv2d(out_channel, out_channel, 4, 2, 1)

    def forward(self, input):
        out_temp_conv = self.conv(input)
        out_temp_skip = self.skip(input)
        print("Conv output shape:", out_temp_conv.shape)
        print("Skip output shape:", out_temp_skip.shape)

        out_temp = torch.relu(out_temp_conv + out_temp_skip)
        out = self.downsample(out_temp)
        return out, out_temp
    
    
class UpBlock(nn.Module):
    def __init__(self, in_channel=128, out_channel=64):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel + in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            BCSA(out_channel, out_channel) 
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            BCSA(out_channel, out_channel)
        )

        self.skip = nn.Conv2d(out_channel + in_channel, out_channel, 1, 1, 0)

    def forward(self, input, FB_in):
        out_temp = self.upsample(input) 
        out_temp = torch.cat([out_temp, FB_in], dim=1) 
        out = self.conv1(out_temp) + self.skip(out_temp) 
        out = self.conv2(out)  
        out = self.conv3(out)  
        return out

    
    
    
class EncodingBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel=512):
        super().__init__()


        self.conv = nn.Sequential(

                                    nn.Conv2d(out_channel, out_channel,3, 1,1),
                                    nn.InstanceNorm2d(out_channel),
                                    nn.ReLU()
                                      )

        self.skip = nn.Conv2d(in_channel,out_channel,1,1,0)
        # self.ca_block = CA_Block(out_channel)
    def forward(self, input):
        out = self.conv(input) + self.skip(input)
        # out = self.ca_block(out)
        return out



class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


        
class UNet(nn.Module):
    def __init__(self, ngf=16, input_channel=3, output_channel=3):
        super(UNet, self).__init__()

        self.init = EncodingBlock(input_channel, ngf)
        self.down1 = DownBlock(ngf, ngf)
        self.down2 = DownBlock(ngf, 2 * ngf)
        self.down3 = DownBlock(2 * ngf, 4 * ngf)


    def forward(self, x, use_sigmoid=True):
        x_init = self.init(x)
        d1, d1_f = self.down1(x_init)
        d2, d2_f = self.down2(d1)
        d3, d3_f = self.down3(d2)

        h = self.encoding(d3)
        hu3 = self.up3(h, d3_f)
        hu2 = self.up2(hu3, d2_f)
        hu1 = self.up1(hu2, d1_f)

        h_out = self.out(torch.cat([hu1, x_init], dim=1))
        h_out = self.conv_fin(h_out)

        if use_sigmoid:
            h_out = self.sigmoid(h_out)

        return h_out

    
    
    

class DoubleUNetWithSimSPPF(nn.Module):
    def __init__(self, ngf=16, input_channel=3, output_channel=3):


        # First UNet
        self.unet1 = UNet(ngf, input_channel, output_channel)

        # SimSPPF layer
        self.sim_sppf = SimSPPF(output_channel, ngf)
        
        # Second UNet
        self.unet2 = UNet(ngf, ngf, output_channel)  # Change input_channel to ngf


    def forward(self, x, use_sigmoid=True):
        
        x = x.to(self.unet1.conv_fin.weight.device)
        # Forward pass through the first UNet
        out1 = self.unet1(x, use_sigmoid)

        # Forward pass through the SimSPPF layer
        spp_out = self.sim_sppf(out1) 

        # Forward pass through the second UNet
        out2 = self.unet2(spp_out, use_sigmoid)
        return out2
        



# Instantiate DoubleUNetWithSimSPPF
double_unet_with_sim_sppf = DoubleUNetWithSimSPPF()


# Example usage
input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
output = double_unet_with_sim_sppf(input_tensor)
print(output.shape)  # Example output shape
