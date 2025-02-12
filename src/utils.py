from matplotlib import pyplot as plt
import torch
from dataset import MnistDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


def show_image(image: torch.Tensor):
    plt.imshow(image, cmap="gray")
    plt.show()


def show_image_in_dataset():
    dataset = MnistDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    image, _, _ = next(iter(dataloader))

    # remove batch size dimension
    image = image.squeeze(0)

    show_image(image)
