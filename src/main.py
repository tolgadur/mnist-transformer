from dataset import MnistDataset
import matplotlib.pyplot as plt


def main():
    dataset = MnistDataset()
    dataset_iterator = iter(dataset)
    example_images, example_labels = next(dataset_iterator)

    plt.imshow(example_images, cmap="gray")
    plt.title(f"Label: {example_labels}")
    plt.show()

    print(example_labels)


if __name__ == "__main__":
    main()
