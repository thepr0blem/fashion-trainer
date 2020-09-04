import fire
from fashion_trainer.trainer import Trainer


def train_network(
    train_csv: str = "data/fashion-mnist_train.csv",
    test_csv: str = "data/fashion-mnist_test.csv",
    model_dir: str = "models/model.pt",
    learning_rate: float = 0.001,
    num_epochs: int = 20,
    batch_size: int = 100,
) -> None:
    """Train CNN for image classification.

    Train Convolutional Neural Network for zalando fashion dataset
    image classification.

    Args:
        train_csv (str): Training data path (.csv). Defaults to "data/fashion-mnist_train.csv"
        test_csv (str): Test data path (.csv). Defaults to "data/fashion-mnist_test.csv"
        model_dir (str): Path to save model. Defaults to "models/model.pt"
        learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.001.
        num_epochs (int): Number of training epochs. Defaults to 20.
        batch_size (int): Batch size. Defaults to 100.

    Returns:
        None
    """
    trainer = Trainer(
        train_csv=train_csv,
        test_csv=test_csv,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    trainer.train()
    trainer.print_per_class_accuracy()
    trainer.save_training_metric_plots()
    trainer.save_model(model_path=model_dir)


def main():
    fire.Fire(train_network)


if __name__ == "__main__":
    main()
