from dataset import MnistDataset
import matplotlib.pyplot as plt


def main():
    dataset = MnistDataset()
    dataset_iterator = iter(dataset)
    example_image, example_label = next(dataset_iterator)
    plt.imshow(example_image, cmap="gray")
    plt.title(f"Label: {example_label}")
    plt.show()


if __name__ == "__main__":
    main()
