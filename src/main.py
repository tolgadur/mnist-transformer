from dataset import MnistDataset


def main():
    dataset = MnistDataset()
    dataset_iterator = iter(dataset)
    example_images, example_labels = next(dataset_iterator)

    print(example_images)
    print(example_labels)


if __name__ == "__main__":
    main()
