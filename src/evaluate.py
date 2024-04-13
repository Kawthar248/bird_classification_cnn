import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.data_loader import prepare_dataframes
from src.model import build_model
import os

def evaluate(model_path: str, test_dir: str) -> None:
    """
    Evaluates the trained PyTorch model on the test dataset.

    Args:
        model_path (str): Path to the saved PyTorch model's state dictionary.
        test_dir (str): Path to the test directory.
    """
    # Set device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model's state dict
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate test accuracy
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    evaluate(model_path='models/bird_classification_model.pth', test_dir=os.path.join('datadata', 'test'))
