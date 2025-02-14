import torch
from dataset import MnistSingleDigitDataset, MnistDataset
from classifier import ClassificationModel
from config import DEVICE, VOCAB
import tqdm
from transformer import Transformer
import matplotlib.pyplot as plt
from utils import load_transformer_model


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


def example_transformer_inference(seed: int = None):
    mnist_dataset = MnistDataset(train=False, num_samples=10000, seed=seed)
    dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1)
    datapoint = next(iter(dataloader))
    image, flattened_patches, input_seq, target_seq = datapoint

    model = load_transformer_model()
    predictions = predict_with_transformer(flattened_patches, model)

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 3)

    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.squeeze(0), cmap="gray")
    ax1.set_title(f"Input Image\nPrediction: {predictions.cpu().numpy()[0]}")
    ax1.axis("off")

    # Get first encoder self-attention layer weights
    first_encoder_attn = model.encoder.encoder_layers[0].attention
    if first_encoder_attn.attention_weights is not None:
        # Average across batch and heads
        encoder_attn_weights = first_encoder_attn.attention_weights.mean(
            dim=(0, 1)
        ).numpy()
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(encoder_attn_weights, cmap="viridis")
        ax2.set_title("Encoder Self-Attention\n(First Layer)")
        ax2.set_xlabel("Key Position (Image Patches)")
        ax2.set_ylabel("Query Position (Image Patches)")

    # Get first decoder self-attention layer weights
    first_decoder_attn = model.decoder.layers[0].self_attn
    if first_decoder_attn.attention_weights is not None:
        # Average across batch and heads
        decoder_attn_weights = first_decoder_attn.attention_weights.mean(
            dim=(0, 1)
        ).numpy()
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(decoder_attn_weights, cmap="viridis")
        ax3.set_title("Decoder Self-Attention\n(First Layer)")
        ax3.set_xlabel("Key Position")
        ax3.set_ylabel("Query Position")

    # Get first cross-attention layer weights
    first_cross_attn = model.decoder.layers[0].cross_attn
    if first_cross_attn.attention_weights is not None:
        # Average across batch and heads
        cross_attn_weights = first_cross_attn.attention_weights.mean(dim=(0, 1)).numpy()
        ax4 = fig.add_subplot(gs[1, 1])
        im = ax4.imshow(cross_attn_weights, cmap="viridis")
        ax4.set_title("Cross-Attention\n(First Layer)")
        ax4.set_xlabel("Key Position (Image Patches)")
        ax4.set_ylabel("Query Position (Sequence)")

    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Attention Weight")

    plt.tight_layout()
    plt.show()


def predict_with_transformer(flattened_patches: torch.Tensor, model: Transformer):
    start_token = VOCAB["<start>"]
    end_token = VOCAB["<end>"]

    batch_size = flattened_patches.shape[0]
    # Start with just the start token for each sequence in the batch
    curr_seq = torch.tensor([[start_token]] * batch_size, device=DEVICE)

    # Move inputs to device
    flattened_patches = flattened_patches.to(DEVICE)

    max_len = 10  # start + 4 digits + end
    with torch.no_grad():
        while curr_seq.shape[1] < max_len:
            output = model(flattened_patches, curr_seq)
            logits = output[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_seq = torch.cat([curr_seq, next_token], dim=1)

            # Check if all sequences have generated end token
            if (next_token == end_token).all():
                break

        predictions = curr_seq[:, 1:-1]  # Remove start and end tokens

    return predictions


def evaluate_transformer(num_samples: int = 10000):
    model = load_transformer_model()
    model.eval()

    # Load test dataset
    dataset = MnistDataset(train=False, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    correct_sequences = 0
    total_sequences = 0
    correct_digits = 0
    total_digits = 0

    with torch.no_grad():
        for _, flattened_patches, input_seq, target_seq in tqdm.tqdm(
            dataloader, desc="Evaluating transformer model"
        ):
            target_seq = target_seq.to(DEVICE)
            predictions = predict_with_transformer(flattened_patches, model)

            # Compare predictions with target sequence
            # Exclude end token from target_seq
            true_digits = target_seq[:, :-1]  # exclude end token

            # Calculate sequence-level accuracy (must match exactly)
            sequences_correct = (predictions == true_digits).all(dim=1)
            correct_sequences += sequences_correct.sum().item()
            total_sequences += sequences_correct.size(0)

            # Calculate digit-level accuracy
            digits_correct = predictions == true_digits
            correct_digits += digits_correct.sum().item()
            total_digits += digits_correct.numel()

    sequence_accuracy = (correct_sequences / total_sequences) * 100
    digit_accuracy = (correct_digits / total_digits) * 100

    print(f"Transformer Sequence-Level Accuracy: {sequence_accuracy:.2f}%")
    print(f"Transformer Digit-Level Accuracy: {digit_accuracy:.2f}%")
