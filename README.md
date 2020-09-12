# fashion-trainer

Source data: [Kaggle - Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist/data?select=fashion-mnist_train.csvm)

Run with:
```bash
>> poetry install
>> bash data/download_data.sh
>> poetry run fashion-trainer
```


```
NAME
    fashion-trainer - Train CNN for image classification.

SYNOPSIS
    fashion-trainer <flags>

DESCRIPTION
    Train Convolutional Neural Network for zalando fashion dataset
    image classification.

FLAGS
    --train_csv=TRAIN_CSV
        Training data path (.csv). Defaults to "data/fashion-mnist_train.csv"
    --test_csv=TEST_CSV
        Test data path (.csv). Defaults to "data/fashion-mnist_test.csv"
    --model_dir=MODEL_DIR
        Path to save model. Defaults to "models/model.pt"
    --learning_rate=LEARNING_RATE
        Learning rate for Adam optimizer. Defaults to 0.001.
    --num_epochs=NUM_EPOCHS
        Number of training epochs. Defaults to 20.
    --batch_size=BATCH_SIZE
        Batch size. Defaults to 100.

```
