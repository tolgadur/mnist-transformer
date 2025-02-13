from matplotlib import pyplot as plt
import torch
from dataset import MnistDataset


def show_image(image: torch.Tensor, label: int):
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()


def show_image_in_dataset():
    dataset = MnistDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    image, _, input_seq, target_seq = next(iter(dataloader))

    # remove batch size dimension
    image = image.squeeze(0)

    # Show first digit from the sequence (excluding start token)
    label = input_seq[0, 1].item()
    show_image(image, label)
