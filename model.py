import torch.nn as nn


class ConvNetLight(nn.Module):
    def __init__(self):
        super(ConvNetLight, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 7 * 7, 50)
        # self.relu = nn.ReLU()
        self.output = nn.Linear(50, 10)  # Output dimension is 10 for classification

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.relu(self.fc1(x))
        x = self.output(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2, padding=1)
        # self.relu = nn.ReLU()
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 7 * 7, 100)  
        # self.relu = nn.ReLU()
        self.output = nn.Linear(100, 10)  # Output dimension is 10 for classification

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.relu(self.fc1(x))
        x = self.output(x)
        return x


# Define a basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# Define the ResNet-8 model
class ResNet8_3(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet8_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(16, 16, 2)
        self.layer2 = self.make_layer(16, 32, 2, stride=2)
        self.layer3 = self.make_layer(32, 64, 2, stride=2)
        # self.avg_pool = nn.AvgPool2d(8)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Use adaptive average pooling
        self.output = nn.Linear(64, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out


class ConvNetLarge(nn.Module):
    def __init__(self):
        super(ConvNetLarge, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU()
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Assuming input size of
        # self.relu = nn.ReLU()
        self.output = nn.Linear(512, 10)  # Output dimension is 10 for classification

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.bn1(self.maxpool(self.relu(self.conv2(x))))
        x = self.bn2(self.maxpool(self.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.relu(self.fc1(x))
        x = self.output(x)
        return x
