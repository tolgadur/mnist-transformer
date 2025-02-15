# import train
import evals


def main():
    # train.train_transformer(batch_size=512, epochs=20)

    # transformer inference
    evals.example_transformer_inference(seed=42)

    # evaluate transformer
    evals.evaluate_transformer()


if __name__ == "__main__":
    main()
