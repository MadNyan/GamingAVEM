import torch
import torch.nn as nn

class Conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='valid'): 
        super(Conv2d_bn, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Stem_layer(nn.Module):
    def __init__(self, in_channels=1): 
        super(Stem_layer, self).__init__()
        self.conv_0 = Conv2d_bn(in_channels, 32, kernel_size=(3, 3), stride=(2, 2))
        self.conv_1 = Conv2d_bn(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv_2 = Conv2d_bn(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.pool_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv_3 = Conv2d_bn(64, 80, kernel_size=(3, 3), stride=(2, 2))
        self.conv_4 = Conv2d_bn(80, 192, kernel_size=(3, 3), stride=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_0(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_1(x)
        return x

class Inception_1(nn.Module):
    def __init__(self, in_channels=192): 
        super(Inception_1, self).__init__()
        self.branch_0 = Conv2d_bn(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding='same')

        self.branch_1_0 = Conv2d_bn(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_1_1 = Conv2d_bn(64, 96, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.branch_2_0 = Conv2d_bn(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_2_1 = Conv2d_bn(64, 96, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.branch_2_2 = Conv2d_bn(96, 96, kernel_size=(3, 3), stride=(1, 1), padding='same')
        
        self.conv_0 = Conv2d_bn(480, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        branch_0 = self.branch_0(x)

        branch_1 = self.branch_1_0(x)
        branch_1 = self.branch_1_1(branch_1)

        branch_2 = self.branch_2_0(x)
        branch_2 = self.branch_2_1(branch_2)
        branch_2 = self.branch_2_2(branch_2)
        
        mix_0 = torch.cat([x, branch_0, branch_1, branch_2], 1)

        x = self.conv_0(mix_0)

        return x

class Inception_2(nn.Module):
    def __init__(self, in_channels=96): 
        super(Inception_2, self).__init__()
        self.branch_0 = Conv2d_bn(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.branch_1_0 = Conv2d_bn(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_1_1 = Conv2d_bn(96, 128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.branch_1_2 = Conv2d_bn(128, 160, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.branch_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.branch_3 = Conv2d_bn(416, 192, kernel_size=(1, 1), stride=(1, 1), padding='same')

        self.branch_4_0 = Conv2d_bn(416, 128, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_4_1 = Conv2d_bn(128, 160, kernel_size=(7, 1), stride=(1, 1), padding='same')
        self.branch_4_2 = Conv2d_bn(160, 160, kernel_size=(1, 7), stride=(1, 1), padding='same')

    def forward(self, x):
        branch_0 = self.branch_0(x)

        branch_1 = self.branch_1_0(x)
        branch_1 = self.branch_1_1(branch_1)
        branch_1 = self.branch_1_2(branch_1)

        branch_2 = self.branch_2(x)

        mix_0 = torch.cat([x, branch_0, branch_1, branch_2], 1)

        branch_3 = self.branch_3(mix_0)

        branch_4 = self.branch_4_0(mix_0)
        branch_4 = self.branch_4_1(branch_4)
        branch_4 = self.branch_4_2(branch_4)

        mix_1 = torch.cat([x, mix_0, branch_3, branch_4], 1)

        return mix_1

class Translate_layer(nn.Module):
    def __init__(self, in_channels=864): 
        super(Translate_layer, self).__init__()
        self.conv_0 = Conv2d_bn(in_channels, 192, kernel_size=(1, 1), stride=(1, 1))
        self.pool_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn_0 = nn.BatchNorm2d(num_features=192)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.pool_0(x)
        x = self.bn_0(x)

        return x

class Inception_3(nn.Module):
    def __init__(self, in_channels=192): 
        super(Inception_3, self).__init__()
        self.branch_0 = Conv2d_bn(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding='same')

        self.branch_1_0 = Conv2d_bn(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_1_1 = Conv2d_bn(192, 256, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.branch_2_0 = Conv2d_bn(in_channels, 160, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_2_1 = Conv2d_bn(160, 192, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.branch_2_2 = Conv2d_bn(192, 256, kernel_size=(3, 3), stride=(1, 1), padding='same')

        self.branch_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.branch_4 = Inception_block(1088)
        self.branch_5 = Inception_block(1344)

    def forward(self, x):
        branch_0 = self.branch_0(x)

        branch_1 = self.branch_1_0(x)
        branch_1 = self.branch_1_1(branch_1)

        branch_2 = self.branch_2_0(x)
        branch_2 = self.branch_2_1(branch_2)
        branch_2 = self.branch_2_2(branch_2)

        branch_3 = self.branch_3(x)

        mix_0 = torch.cat([x, branch_0, branch_1, branch_2, branch_3], 1)

        branch_4 = self.branch_4(mix_0)
        branch_5 = self.branch_5(branch_4)

        mix_1 = torch.cat([x, mix_0, branch_4, branch_5], 1)

        return mix_1

class Inception_block(nn.Module):
    def __init__(self, in_channels=1088): 
        super(Inception_block, self).__init__()
        self.branch_0 = Conv2d_bn(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding='same')

        self.branch_1_0 = Conv2d_bn(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.branch_1_1 = Conv2d_bn(128, 192, kernel_size=(3, 1), stride=(1, 1), padding='same')
        self.branch_1_2 = Conv2d_bn(192, 256, kernel_size=(1, 3), stride=(1, 1), padding='same')

        self.branch_2 = Conv2d_bn(512, 256, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        branch_0 = self.branch_0(x)

        branch_1 = self.branch_1_0(x)
        branch_1 = self.branch_1_1(branch_1)
        branch_1 = self.branch_1_2(branch_1)

        mix_0 = torch.cat([branch_0, branch_1], 1)

        branch_2 = self.branch_2(mix_0)

        mix_1 = torch.cat([x, branch_2], 1)

        return mix_1

class Dense_FaceLiveNet(nn.Module):
    def __init__(self, num_classes=6):
        super(Dense_FaceLiveNet, self).__init__()
        self.stem_layer = Stem_layer()
        self.inception_1 = Inception_1()
        self.inception_2 = Inception_2()
        self.translate_layer = Translate_layer()
        self.inception_3 = Inception_3()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=4224, out_features=num_classes)
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.stem_layer(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.translate_layer(x)
        x = self.inception_3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x