import torch
import torch.nn as nn
from IPython import embed
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init

@NECKS.register_module()
class ConvMemFusion(nn.Module):
    def __init__(self, out_channels, use_overlap=True, memory_length=4):
        super(ConvMemFusion, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        assert use_overlap
        self.ln = nn.LayerNorm(out_channels)
        self.use_overlap = use_overlap
        self.conv_overlap = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.memory_fusion = nn.Sequential(
            # use kernel size 3 and same padding to keep the spatial size
            nn.Conv2d(out_channels*(memory_length+1)+32, out_channels, kernel_size=3, dilation=2, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding='same', bias=False),
            nn.ReLU()
        )
        self.out_channels = out_channels
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, h, x, overlap=None):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if overlap is not None:
            if len(overlap.shape) == 3:
                overlap = overlap.unsqueeze(0)
            overlap_conv = self.conv_overlap(overlap)
        hoverlapx = torch.cat([h, overlap_conv, x], dim=1) # [1, c*memory_len+32+c, h, w]
        out = self.memory_fusion(hoverlapx).squeeze(0) # [c, h, w]
        out = self.ln(out.permute(1, 2, 0)).permute(2, 0, 1).contiguous()
        return out