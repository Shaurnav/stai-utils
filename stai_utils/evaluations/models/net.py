import torch.nn as nn

h_dims = [32, 64, 64, 128, 128, 256, 256, 512]


class ConvBlockDown(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, norm_type, down_sample=True, dim=2
    ):
        super(ConvBlockDown, self).__init__()
        self.norm_type = norm_type
        self.down_sample = down_sample

        if dim == 2:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.down = nn.MaxPool2d(2)

        elif dim == 3:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()

            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.down = nn.MaxPool3d(2)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.down_sample:
            out = self.down(out)
        return out


class UNetEncoder(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super().__init__()
        self.dim = dim

        self.block1 = ConvBlockDown(input_ch, h_dims[0], 1, norm_type, False, dim)
        self.block2 = ConvBlockDown(h_dims[0], h_dims[1], 1, norm_type, True, dim)

        self.block3 = ConvBlockDown(h_dims[1], h_dims[2], 1, norm_type, False, dim)
        self.block4 = ConvBlockDown(h_dims[2], h_dims[3], 1, norm_type, True, dim)

        self.block5 = ConvBlockDown(h_dims[3], h_dims[4], 1, norm_type, False, dim)
        self.block6 = ConvBlockDown(h_dims[4], h_dims[5], 1, norm_type, True, dim)

        self.block7 = ConvBlockDown(h_dims[5], h_dims[6], 1, norm_type, False, dim)
        # self.block8 = layers.ConvBlockDown(
        #     h_dims[6], h_dims[7], 1, norm_type, True, dim
        # )

        self.block9 = ConvBlockDown(h_dims[6], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block9(out)
        return out
