import train
import utils


def main():
    # utils.show_image_in_dataset()

    train.train_classifier()
    train.train_single_digit_classifier()


if __name__ == "__main__":
    main()
