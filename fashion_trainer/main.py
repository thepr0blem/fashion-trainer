from fashion_trainer.constants import MODEL_DIR, TEST_DATA_DIR, TRAIN_DATA_DIR
from fashion_trainer.trainer import Trainer


def run_training() -> None:
    trainer = Trainer(train_csv=TRAIN_DATA_DIR, test_csv=TEST_DATA_DIR)
    trainer.train()
    trainer.print_per_class_accuracy()
    trainer.save_training_metric_plots()
    trainer.save_model(model_path=MODEL_DIR)


def main():
    run_training()


if __name__ == "__main__":
    main()
