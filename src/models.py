import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_resnet50(num_classes, feature_extract=True):
    """
    Constructs a ResNet50 architecture using transfer learning.
    Transforms output via Softmax probabilties representations:
    $P(y=j \mid x) = \\frac{e^{z_j}}{\\sum_{k=1}^{K} e^{z_k}}$
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    set_parameter_requires_grad(model, feature_extract)
    
    # Custom Head: GlobalAveragePooling -> Dropout(0.4) -> Dense(11)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_mobilenet_v2(num_classes, feature_extract=True):
    """
    Constructs a MobileNetV2 architecture using transfer learning.
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    set_parameter_requires_grad(model, feature_extract)
    
    # Custom Head: Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_efficientnet_b0(num_classes, feature_extract=True):
    """
    Constructs an EfficientNet-B0 architecture using transfer learning.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model, feature_extract)
    
    # Custom Head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )
    return model
