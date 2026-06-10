import torch
import torch.nn as nn

class RowBiLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        # x: [B, C, H, W]  -> [B,256,16,16]
        B, C, H, W = x.shape
        x_row = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_row = x_row.view(B * H, W, C)              # [B*H, W, C]
        out, _ = self.lstm(x_row)                    # [B*H, W, 2*hidden_dim]
        out = self.proj(out)                         # [B*H, W, C]
        out = out.view(B, H, W, C)                   # [B, H, W, C]
        out = out.permute(0, 3, 1, 2)                # [B, C, H, W]
        return out

class ColumnBiLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(2 * hidden_dim, input_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_col = x.permute(0, 3, 2, 1).contiguous()   # [B, W, H, C]
        x_col = x_col.view(B * W, H, C)              # [B*W, H, C]
        out, _ = self.lstm(x_col)                    # [B*W, H, 2*hidden_dim]
        out = self.proj(out)                         # [B*W, H, C]
        out = out.view(B, W, H, C)
        out = out.permute(0, 3, 2, 1)                # [B, C, H, W]
        return out

class SpatialRelationModel(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=256, num_classes=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.row_lstm = RowBiLSTM(input_dim=in_channels, hidden_dim=hidden_dim)
        self.col_lstm = ColumnBiLSTM(input_dim=in_channels, hidden_dim=hidden_dim)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.cnn(x)
        x = x + self.row_lstm(x)
        x = x + self.col_lstm(x)
        out = self.classifier(x)
        return out