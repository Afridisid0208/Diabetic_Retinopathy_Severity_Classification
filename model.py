import torch.nn as nn
import timm

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetClassifier, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
