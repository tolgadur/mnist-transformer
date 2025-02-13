import evals
from dataset import MnistDataset
import train


def main():
    train.train_single_digit_classifier()
    evals.evaluate_single_digit_classifier()


if __name__ == "__main__":
    main()
