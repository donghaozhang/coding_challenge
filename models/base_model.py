import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import convnext_base, convnext_tiny, convnext_small, convnext_large


class BaseModel(nn.Module):
    def __init__(self, model_type='tiny'):
        super().__init__()
        if model_type == 'tiny':
            self.backbone = convnext_tiny()
            self.linear_out = 768
        elif model_type == 'small':
            self.backbone = convnext_small()
            self.linear_out = 768
        elif model_type == 'base':
            self.backbone = convnext_base()
            self.linear_out = 1024
        elif model_type == 'large':
            self.backbone = convnext_large()
            self.linear_out = 1536

        # Remove the last layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.backbone.add_module('flatten', nn.Flatten())
        self.classifier = nn.Linear(self.linear_out, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def get_feature(self, x):
        x = self.backbone(x)
        return x

    def classify(self, x):
        x = self.classifier(x)
        return x

    @staticmethod
    def compute_loss(logits, labels):
        return F.cross_entropy(logits, labels)


if __name__ == '__main__':
    model = BaseModel(model_type='large')
    image_input = torch.randn(1, 3, 64, 64)
    output = model(image_input)
    print(output.shape)
