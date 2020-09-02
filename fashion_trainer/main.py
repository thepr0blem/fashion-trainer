from fashion_trainer.constants import MODEL_DIR, TEST_DATA_DIR, TRAIN_DATA_DIR
from fashion_trainer.trainer import Trainer


def run_training():
    trainer = Trainer(train_csv=TRAIN_DATA_DIR, test_csv=TEST_DATA_DIR)
    trainer.train()
    trainer.save_model(MODEL_DIR)


def main():
    run_training()


if __name__ == "__main__":
    main()
