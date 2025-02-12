import torch
from dataset import MnistDataset
import tqdm
from utils import DEVICE
from classifier import ClassificationModel
import torch.nn as nn


def train(epochs: int = 10, batch_size: int = 512, lr: float = 0.001):
    # define model
    model = ClassificationModel().to(DEVICE)

    # load dataset
    dataset = MnistDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for _, flattened, labels in tqdm.tqdm(dataloader):
            flattened = flattened.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(flattened)
            loss = criterion(logits, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/classifier_model.pth")
    print("Model saved as classifier_model.pth")
