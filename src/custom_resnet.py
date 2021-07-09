import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary.torchsummary import summary

# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
class CustomBasicBlock(nn.Module):
    """Resnet Basic block custom architecture

    Args:
        nn (nn.Module): Extends nn.Module class
    """
    expansion=1
    def __init__(self, in_planes, out_planes, stride=1, padding=1):
        super(CustomBasicBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=padding),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False))

        self.block_res = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=2, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x_out = self.block1(x)
        out = self.block_res(x)
        out += x_out
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(CustomResNet, self).__init__()
        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
        )
        
        self.layer2 = self._make_layer(block, in_planes=64, out_planes=128, stride=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = self._make_layer(block, in_planes=256, out_planes=512, stride=1)

        self.pool = nn.MaxPool2d(4,2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, in_planes, out_planes, stride):
        strides = [stride] # + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, out_planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ModelLoader():
    def modelsummary(inputsize):
        use_cuda = torch.cuda.is_available()
        device = 'cuda:0' if use_cuda else 'cpu'
        model = CustomResNet(CustomBasicBlock).to(device)
        return model, summary(model, input_size=inputsize)
