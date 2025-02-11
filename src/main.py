from dataset import MnistDataset
import matplotlib.pyplot as plt


def main():
    dataset = MnistDataset()
    dataset_iterator = iter(dataset)
    images, vector, labels = next(dataset_iterator)

    plt.imshow(images)
    plt.show()


if __name__ == "__main__":
    main()
