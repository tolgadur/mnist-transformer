import train
import utils
import evals


def main():
    # utils.show_image_in_dataset()

    train.train_classifier()
    train.train_single_digit_classifier()

    evals.evaluate_single_digit_classifier()
    evals.evaluate_classifier()


if __name__ == "__main__":
    main()
