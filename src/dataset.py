import torch
import torchvision
import torchvision.transforms as transforms


class MnistDataset(torch.utils.data.IterableDataset):
    def __init__(self, train=True):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ]
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
            top = torch.cat((images[0, 0], images[1, 0]), dim=1)
            bottom = torch.cat((images[2, 0], images[3, 0]), dim=1)
            combined = torch.cat((top, bottom), dim=0)

            # Reshape from (4, 28, 28) to (4, 784)
            flattened = combined.reshape(4, -1)
            yield flattened, labels
