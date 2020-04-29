import torchvision.models as models
import torch.nn as nn
import torch

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(1024,234)

adjustmentnet = models.densenet121(pretrained=True)
adjustmentnet.classifier = nn.Linear(1024,136)

if __name__ == '__main__':

    x = torch.rand((32,3,224,224))
    y_pred = densenet(x)

