from matplotlib import pyplot as plt
import torch
from dataset import MnistDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "<start>": 10,
    "<end>": 11,
}


def show_image(image: torch.Tensor, label: int):
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()


def show_image_in_dataset():
    dataset = MnistDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    image, _, label = next(iter(dataloader))

    # remove batch size dimension
    image = image.squeeze(0)

    show_image(image, label)
