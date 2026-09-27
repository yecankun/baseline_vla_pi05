import torch
from torchsummary import summary

from torch.nn import *
from torchvision.models import *

class OneNet(Module):

    def __init__(self, backbone_name, function_name):
        super().__init__()
        self.end2end = Sequential(
            eval(backbone_name)(),
            eval(function_name)(),
            Linear(1000, 512),
            eval(function_name)(),
            Linear(512, 256),
            eval(function_name)(),
            Linear(256, 128),
            eval(function_name)(),
            Linear(128, 64),
            eval(function_name)(),
            Linear(64, 1),
            )

    def forward(self, x):
        return self.end2end(x)
    
if __name__ == "__main__":
    
    net = OneNet("resnet18", "GELU").to("cuda")
    summary(net, (3, 224, 224))