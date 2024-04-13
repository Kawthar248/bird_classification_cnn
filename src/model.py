import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torch.utils.tensorboard import SummaryWriter


def build_model(num_classes: int = 525) -> nn.Module:
    """
    Builds a CNN model using the EfficientNetB0 architecture with pre-trained weights and custom layers.

    Args:
        num_classes (int): Number of output classes. Default is 525.

    Returns:
        nn.Module: The CNN model.
    """
    # Load the pre-trained EfficientNetB0 model
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    pretrained_model = models.efficientnet_b0(weights=weights)

    # Set the pre-trained model to non-trainable
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Custom layers on top of the pre-trained model
    num_features = pretrained_model.classifier[1].in_features
    pretrained_model.classifier = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(256, num_classes),
        nn.Softmax(dim=1)
    )

    return pretrained_model


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """
    Creates an optimizer for the model.

    Returns:
        optim.Optimizer: The Adam optimizer.
    """
    return optim.Adam(model.parameters(), lr=0.0001)


def get_loss_function() -> nn.Module:
    """
    Creates a loss function for the model.

    Returns:
        nn.Module: The categorical cross-entropy loss function.
    """
    return nn.CrossEntropyLoss()


def get_callbacks(log_dir: str = "runs/birds_classification") -> SummaryWriter:
    """
    Creates a TensorBoard writer for logging training metrics.

    Returns:
        SummaryWriter: A TensorBoard writer.
    """
    return SummaryWriter(log_dir=log_dir)
