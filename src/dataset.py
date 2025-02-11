import torch
import torchvision
import torchvision.transforms as transforms


class MnistDataset(torch.utils.data.IterableDataset):
    def __init__(self, train=True):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=4, shuffle=True
        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for images, labels in self.dataloader:
            top = torch.cat(
                (images[0, 0], images[1, 0]), dim=1
            )  # images[0, 0] has dim 28 x 28 i.e. height and width
            bottom = torch.cat((images[2, 0], images[3, 0]), dim=1)
            yield torch.cat((top, bottom), dim=0), labels[0]
