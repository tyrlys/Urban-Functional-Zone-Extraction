import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)
        return self.sigmoid(attention) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class MiniResNet(nn.Module):
    def __init__(self, in_channels=6, base_channels=32, out_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(base_channels, base_channels),
            BasicBlock(base_channels, base_channels)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(base_channels, base_channels * 2, stride=2),
            BasicBlock(base_channels * 2, base_channels * 2)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base_channels * 2, base_channels * 4, stride=2),
            BasicBlock(base_channels * 4, base_channels * 4)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(base_channels * 4, out_channels, stride=2),
            BasicBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.stem(x)      # [B,32,32,32]
        x = self.layer1(x)    # [B,32,32,32]
        x = self.layer2(x)    # [B,64,16,16]
        x = self.layer3(x)    # [B,128,8,8]
        x = self.layer4(x)    # [B,128,4,4]
        return x

def build_backbone(in_channels, out_channels=128):
    return MiniResNet(in_channels, base_channels=32, out_channels=out_channels)

class CrossModalFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_v_to_attn = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_s_to_attn = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, F_v, F_s):
        A_s = torch.sigmoid(self.conv_v_to_attn(F_v))
        F_s_prime = F_s * A_s
        A_v = torch.sigmoid(self.conv_s_to_attn(F_s))
        F_v_prime = F_v * A_v
        F_fusion = torch.cat([F_v_prime, F_s_prime], dim=1)
        return F_fusion