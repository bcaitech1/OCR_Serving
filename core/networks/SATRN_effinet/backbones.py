import timm

import torch
import torch.nn as nn


class Shallow_cnn(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(Shallow_cnn, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(hidden_dim // 2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(hidden_dim)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool_1(self.bn_1(self.conv_1(x)))
        out = self.pool_2(self.bn_2(self.conv_2(out)))

        return out


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))

        return self.pool(out)


class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout=0.2, num_bn=3):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_size, inter_size, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)

        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [BottleneckBlock(input_size + i * growth_rate, growth_rate, dropout=dropout) for i in range(depth)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """

    def __init__(self, in_channels, num_in_features, hidden_dim, dropout=0.2, depth=16, growth_rate=24):
        super(DeepCNN300, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, num_in_features, kernel_size=7, stride=2, padding=3, bias=False,)
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        num_features = num_in_features
        self.block1 = DenseBlock(num_features, growth_rate=growth_rate, depth=depth, dropout=dropout,)

        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)

        num_features = num_features // 2
        self.block2 = DenseBlock(num_features, growth_rate=growth_rate, depth=depth, dropout=dropout,)

        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(num_features, hidden_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        out = self.max_pool(self.relu(self.norm0(self.conv0(input))))

        out = self.block1(out)

        out = self.trans1(out)

        out = self.block2(out)

        out_A = self.trans2_conv(self.trans2_relu(self.trans2_norm(out)))

        return out_A


class efficientnet_backbone(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout):
        super(efficientnet_backbone, self).__init__()

        self.in_channels = in_channels
        if in_channels == 1:
            self.converter = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

        self.efficient_net = timm.create_model("efficientnetv2_m", features_only=True)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dropout = nn.Dropout(p=dropout)

        self.upsampling_1 = nn.Sequential(
            nn.Conv2d(512 + 176, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.upsampling_2 = nn.Sequential(
            nn.Conv2d(512 + 80, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsampling_3 = nn.Sequential(
            nn.Conv2d(256 + 48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.upsampling_4 = nn.Sequential(
            nn.Conv2d(128 + 24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.trans_conv = nn.Sequential(
            nn.Conv2d(64, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.in_channels == 1:
            raise Exception("efficientnet_backbone supports only 3 channels.")

        features = self.efficient_net(x)
        features.reverse()

        upsampled_1 = self.upsample(features[0])
        upsampled_1 = torch.cat([upsampled_1, features[1]], dim=1)
        upsampled_1 = self.upsampling_1(upsampled_1)
        upsampled_1 = self.dropout(upsampled_1)

        upsampled_2 = self.upsample(upsampled_1)
        upsampled_2 = torch.cat([upsampled_2, features[2]], dim=1)
        upsampled_2 = self.upsampling_2(upsampled_2)
        upsampled_2 = self.dropout(upsampled_2)

        upsampled_3 = self.upsample(upsampled_2)
        upsampled_3 = torch.cat([upsampled_3, features[3]], dim=1)
        upsampled_3 = self.upsampling_3(upsampled_3)
        upsampled_3 = self.dropout(upsampled_3)

        upsampled_4 = self.upsample(upsampled_3)
        upsampled_4 = torch.cat([upsampled_4, features[4]], dim=1)
        upsampled_4 = self.upsampling_4(upsampled_4)
        upsampled_4 = self.dropout(upsampled_4)

        out = self.trans_conv(upsampled_4)
        out = self.dropout(out)

        return out
