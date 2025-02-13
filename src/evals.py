import torch
from dataset import MnistSingleDigitDataset
from classifier import ClassificationModel
from config import DEVICE
import tqdm


def evaluate_single_digit_classifier():
    # Load model
    model = ClassificationModel(seq_len=4).to(DEVICE)
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
