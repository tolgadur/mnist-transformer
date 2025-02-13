import torch
import torchvision
import torchvision.transforms as transforms
from config import VOCAB


class MnistSingleDigitDataset(torch.utils.data.Dataset):
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

    def __len__(self):
        return len(self.dataset)

    def _create_patches(self, image: torch.Tensor, patch_dim: int):
        patch_size = image.shape[0] // patch_dim  # 56 / 4 = 14

        image = image.unfold(0, patch_size, patch_size)  # shape: ( 4 x 56 x 14)
        image = image.unfold(1, patch_size, patch_size)  # shape: ( 4 x 4 x 14 x 14)

        patches = image.reshape(
            shape=(-1, patch_size, patch_size)
        )  # shape: (16 x 14 x 14)

        return patches

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.squeeze(0)  # remove channel dimension, shape: (56 x 56)

        # Create patches and flatten them
        patches = self._create_patches(image, 2)  # shape: (4, 14, 14)
        flattened = patches.reshape(4, -1)  # shape: (4, 196)

        return image, flattened, label


class MnistDataset(torch.utils.data.IterableDataset):
    def __init__(self, train=True, num_samples=500000, seed=42):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),
            ]
        )
        self.dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
        self.num_samples = num_samples

        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Pre-generate all indices for the entire dataset
        self.indices = torch.randint(
            len(self.dataset), size=(num_samples, 4), generator=generator
        )

    def __len__(self):
        return self.num_samples

    def _create_patches(self, image: torch.Tensor, patch_dim: int):
        patch_size = image.shape[0] // patch_dim  # 56 / 4 = 14

        image = image.unfold(0, patch_size, patch_size)  # shape: ( 4 x 56 x 14)
        image = image.unfold(1, patch_size, patch_size)  # shape: ( 4 x 4 x 14 x 14)

        patches = image.reshape(
            shape=(-1, patch_size, patch_size)
        )  # shape: (16 x 14 x 14)

        return patches

    def _arrange_images(self, images: torch.Tensor) -> torch.Tensor:
        # Keep channel dimension when concatenating. The image has dimensions
        # (batch_size, channels, height, width) or (4 x 1 x 28 x 28) in this case.
        top = torch.cat((images[0], images[1]), dim=2)  # concat along width
        bottom = torch.cat((images[2], images[3]), dim=2)
        combined = torch.cat((top, bottom), dim=1)  # concat along height
        # combined shape is now (1 x 56 x 56)

        return combined

    def __iter__(self):
        """
        Yields:
            combined: shape: (1 x 56 x 56) - 4 images combined into one
            flattened: shape: (16 x 196) - 16 patches flattened into vector
            input_seq: shape: (5) - Four indices with <start> token prepended
            target_seq: shape: (5) - Four indices with <end> token appended
        """
        for sample_indices in self.indices:
            # Get the images and labels for these indices using list comprehension
            images = [self.dataset[idx.item()][0] for idx in sample_indices]
            labels = [self.dataset[idx.item()][1] for idx in sample_indices]

            # Stack the images and labels
            images = torch.stack(images)
            labels = torch.tensor(labels)

            # Create input sequence with <start> token prepended
            input_seq = torch.cat([torch.tensor([VOCAB["<start>"]]), labels])

            # Create target sequence with <end> token appended
            target_seq = torch.cat([labels, torch.tensor([VOCAB["<end>"]])])

            combined = self._arrange_images(images)

            # remove channel dimension
            combined = combined.squeeze(0)  # shape: (56 x 56)

            # split into 16 images of 14 x 14, using the spatial dimensions only
            patches = self._create_patches(combined, 4)

            # Reshape from (16 x 14 x 14) to (16 x 196)
            flattened_patches = patches.reshape(16, -1)

            yield combined, flattened_patches, input_seq, target_seq
