import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feat_dim = in_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, landmarks=None, return_features=False):
        del landmarks
        features = self.backbone(x)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits

    def get_param_groups(self, backbone_lr, head_lr):
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.dropout.parameters()) + list(self.classifier.parameters())
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]


def build_model(num_classes=7, pretrained=True, dropout=0.4, **kwargs):
    if pretrained:
        print('ResNet18Baseline ignores pretrained=True and always uses weights=None.')
    return ResNet18Baseline(num_classes=num_classes, dropout=dropout)
