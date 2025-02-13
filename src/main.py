import evals
import train


def main():
    train.train_transformer(batch_size=256, epochs=10)


if __name__ == "__main__":
    main()
