import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """(conv => GN => LeakyReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2, bias=False)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1) # skip-connection
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class YNet(nn.Module):
    def __init__(self, n_dim=32):
        super(YNet, self).__init__()
        # RGB branch
        self.inc_x = inconv(3, 64)
        self.down1_x = down(64, 128)
        self.down2_x = down(128, 256)
        self.down3_x = down(256, 256)
        # self.down4_x = down(512, 512)

        # Flow branch
        self.inc_f = inconv(2, 64)
        self.down1_f = down(64, 128)
        self.down2_f = down(128, 256)
        self.down3_f = down(256, 256)
        # self.down4_f = down(512, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        self.outc = outconv(64, n_dim)

        # Foreground mask generation
        self.mask_conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(n_dim, 8, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, f):
        # Encoder
        x1 = self.inc_x(x) # 64
        x2 = self.down1_x(x1) # 128
        x3 = self.down2_x(x2) # 256
        x4 = self.down3_x(x3) # 256

        f1 = self.inc_f(f) # 64
        f2 = self.down1_f(f1) # 128
        f3 = self.down2_f(f2) # 256
        f4 = self.down3_f(f3) # 256

        # Mid-level concat
        y1 = torch.cat([x1, f1], dim=1) # 128
        y2 = torch.cat([x2, f2], dim=1) # 256
        y3 = torch.cat([x3, f3], dim=1) # 512
        y4 = torch.cat([x4, f4], dim=1) # 512

        # Decoder
        y = self.up1(y4, y3) # 256
        y = self.up2(y, y2) # 128
        y = self.up3(y, y1) # 64
        y = self.outc(y)

        # Get Mask
        mask = self.mask_conv(y)
        # mask = nn.functional.softmax(mask, dim=1)[:,:1]
        mask = self.sigmoid(mask)

        return y, mask

class YNetPlus(nn.Module):
    def __init__(self, n_dim=32):
        super(YNetPlus, self).__init__()
        # RGB branch
        self.inc_x = inconv(3, 64)
        self.down1_x = down(64, 128)
        self.down2_x = down(128, 256)
        self.down3_x = down(256, 512)
        self.down4_x = down(512, 512)

        # Flow branch
        self.inc_f = inconv(2, 64)
        self.down1_f = down(64, 128)
        self.down2_f = down(128, 256)
        self.down3_f = down(256, 512)
        self.down4_f = down(512, 512)

        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.outc = outconv(64, n_dim)

        # Foreground mask generation
        # self.mask_conv = nn.Sequential(
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(n_dim, 8, 1, bias=False),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(8, 2, 1, bias=False),
        # )
        self.mask_conv = nn.Conv2d(n_dim, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, f):
        # Encoder
        x1 = self.inc_x(x) # 64
        x2 = self.down1_x(x1) # 128
        x3 = self.down2_x(x2) # 256
        x4 = self.down3_x(x3) # 512
        x5 = self.down4_x(x4) # 512

        f1 = self.inc_f(f) # 64
        f2 = self.down1_f(f1) # 128
        f3 = self.down2_f(f2) # 256
        f4 = self.down3_f(f3) # 512
        f5 = self.down4_f(f4) # 512

        # Mid-level concat
        y1 = torch.cat([x1, f1], dim=1) # 128
        y2 = torch.cat([x2, f2], dim=1) # 256
        y3 = torch.cat([x3, f3], dim=1) # 512
        y4 = torch.cat([x4, f4], dim=1) # 1024
        y5 = torch.cat([x5, f5], dim=1) # 1024

        # Decoder
        y = self.up1(y5, y4) # 512
        y = self.up2(y, y3) # 256
        y = self.up3(y, y2) # 128
        y = self.up4(y, y1) # 64
        y = self.outc(y) # n_dim

        # Get Mask
        mask = self.mask_conv(y)
        # mask = nn.functional.softmax(mask, dim=1)[:,:1]
        mask = self.sigmoid(mask)

        return y, mask

if __name__ == "__main__":
    img, flow = torch.rand((1, 3, 320, 512)), torch.rand((1, 2, 320, 512))
    net = YNet()
    feature, mask = net(img, flow)
    print(sum([x.numel() for x in net.parameters()]))
    print(feature.shape)