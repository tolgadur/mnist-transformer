import torch
from dataset import MnistSingleDigitDataset, MnistDataset
from classifier import ClassificationModel
from utils import DEVICE
import tqdm


def evaluate_single_digit_classifier():
    # Load model
    model = ClassificationModel(num_classes=10, seq_len=4).to(DEVICE)
    model.load_state_dict(torch.load("models/classifier_single_digit_model.pth"))
    model.eval()

    # Load test dataset
    dataset = MnistSingleDigitDataset(train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad():
        for _, flattened, labels in tqdm.tqdm(
            dataloader, desc="Evaluating single digit model"
        ):
            flattened = flattened.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(flattened)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"Single Digit Classifier Accuracy: {accuracy:.2f}%")


def evaluate_classifier():
    # Load model
    model = ClassificationModel(num_classes=9999, seq_len=16).to(DEVICE)
    model.load_state_dict(torch.load("models/classifier_model.pth"))
    model.eval()

    # Load test dataset
    dataset = MnistDataset(
        train=False, num_samples=10000
    )  # Using 10k samples for testing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    correct = 0
    total = 0

    with torch.no_grad():
        for _, flattened, labels in tqdm.tqdm(
            dataloader, desc="Evaluating multi-digit model"
        ):
            flattened = flattened.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(flattened)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"Multi-Digit Classifier Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    print("Evaluating Single Digit Classifier...")
    evaluate_single_digit_classifier()
    print("\nEvaluating Multi-Digit Classifier...")
    evaluate_classifier()
