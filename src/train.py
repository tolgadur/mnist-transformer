import torch
from dataset import MnistSingleDigitDataset, MnistDataset
import tqdm
from config import DEVICE
from classifier import ClassificationModel
from transformer import Transformer
from encoder import Encoder
from decoder import Decoder
import torch.nn as nn


def validate(model, val_dataloader, criterion):
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for _, flattened_patches, input_seq, target_seq in val_dataloader:
            flattened_patches = flattened_patches.to(DEVICE)
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)

            logits = model(flattened_patches, input_seq)

            # Reshape logits and target for cross entropy loss
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            target_seq = target_seq.reshape(-1)

            loss = criterion(logits, target_seq)
            total_val_loss += loss.item()
            num_val_batches += 1

    return total_val_loss / num_val_batches


def train_transformer(epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    # define model
    encoder = Encoder(dropout=0.1, d_model=64, heads=4, use_cls_token=False)
    decoder = Decoder(dropout=0.1, d_model=64, heads=4)
    transformer = Transformer(encoder, decoder).to(DEVICE)

    # load dataset
    dataset = MnistDataset(train=True, num_samples=500000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # validation dataset
    val_dataset = MnistDataset(train=False, num_samples=10000)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        transformer.train()
        total_loss = 0
        num_batches = 0

        for _, flattened_patches, input_seq, target_seq in tqdm.tqdm(dataloader):
            flattened_patches = flattened_patches.to(DEVICE)
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)

            logits = transformer(flattened_patches, input_seq)

            # Reshape logits and target for cross entropy loss
            # logits shape: (batch_size, seq_len, vocab_size)
            # target shape: (batch_size * seq_len, vocab_size)
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            target_seq = target_seq.reshape(-1)

            loss = criterion(logits, target_seq)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        avg_val_loss = validate(transformer, val_dataloader, criterion)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        torch.save(
            transformer.state_dict(),
            f"models/transformer/transformer_model_{epoch}.pth",
        )

    torch.save(transformer.state_dict(), "models/transformer_model.pth")
    print("Model saved as transformer_model.pth")


def train_single_digit_classifier(
    epochs: int = 10, batch_size: int = 32, lr: float = 0.001
):
    # define model
    model = ClassificationModel(seq_len=4).to(DEVICE)

    # load dataset
    dataset = MnistSingleDigitDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for _, flattened_patches, labels in tqdm.tqdm(dataloader):
            flattened_patches = flattened_patches.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(flattened_patches)
            loss = criterion(logits, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/classifier_single_digit_model.pth")
    print("Model saved as classifier_single_digit_model.pth")
