# import a pretrained model from torchvision.models and modify it to fit our needs


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.model = models.resnet18(pretrained=self.pretrained)

        # Modify the last linear layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)

    

class ResNet34():
    def __init__(self, num_classes, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.model = models.resnet34(pretrained=self.pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
    def forward(self,x):
        return self.model(x)
    

class ResNet50():
    def __init__(self, num_classes, pretrained=True):
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.model = models.resnet50(pretrained=self.pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    def forward(self,x):
        return self.model(x)