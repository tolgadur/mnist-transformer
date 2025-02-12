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

    def _create_patches(self, image: torch.Tensor, patch_dim: int):
        patch_size = image.shape[0] // patch_dim  # 56 / 4 = 14

        image = image.unfold(0, patch_size, patch_size)  # shape: ( 4 x 56 x 14)
        image = image.unfold(1, patch_size, patch_size)  # shape: ( 4 x 4 x 14 x 14)

        patches = image.reshape(
            shape=(-1, patch_size, patch_size)
        )  # shape: (16 x 14 x 14)

        return patches

    def _labels_to_class_index(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert 4 digit labels into a single class index between 0-9999."""
        # labels shape: (4,)
        # Convert to a single number: e.g., [1,2,3,4] -> 1234
        return labels[0] * 1000 + labels[1] * 100 + labels[2] * 10 + labels[3]

    def __iter__(self):
        """
        Yields:
            combined: shape: (1 x 56 x 56) - These are the 4 images combined into one, with channel dim
            flattened: shape: (16 x 196) - These are the 16 patches flattened into a vector
            class_index: shape: (1) - Single class index between 0-9999 representing the 4 digits
        """
        for images, labels in self.dataloader:
            # Keep channel dimension when concatenating. The image has dimensions
            # (batch_size, channels, height, width) or (4 x 1 x 28 x 28) in this case.
            top = torch.cat((images[0], images[1]), dim=2)  # concat along width
            bottom = torch.cat((images[2], images[3]), dim=2)
            combined = torch.cat((top, bottom), dim=1)  # concat along height
            # combined shape is now (1 x 56 x 56)

            # remove channel dimension
            combined = combined.squeeze(0)  # shape: (56 x 56)

            # split into 16 images of 14 x 14, using the spatial dimensions only
            patches = self._create_patches(combined, 4)

            # Reshape from (16 x 14 x 14) to (16 x 196)
            flattened = patches.reshape(16, -1)

            # Convert labels to class index
            labels = self._labels_to_class_index(labels)

            yield combined, flattened, labels
