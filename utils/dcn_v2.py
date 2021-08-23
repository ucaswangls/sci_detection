from torchvision.ops import DeformConv2d
from torch import nn 
import torch

class DCN_V2(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1,groups=1):
        super(DCN_V2, self).__init__()
        self.offset_channel = 2*kernel_size**2
        self.mask_channel = 1*kernel_size**2
        channels_ = groups * 3 * kernel_size**2 
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          channels_,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=True)
        self.dcn_v2_conv = DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        offset, mask = torch.split(out,split_size_or_sections=[self.offset_channel,self.mask_channel],dim=1) 
        mask = torch.sigmoid(mask)
        return self.dcn_v2_conv(input, offset, mask)