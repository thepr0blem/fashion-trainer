import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fashion_trainer.constants import TEST_DATA_DIR, TRAIN_DATA_DIR
from fashion_trainer.dataset import FashionDataset
from fashion_trainer.model import FashionCNN
from torch.autograd import Variable
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_csv = pd.read_csv(TRAIN_DATA_DIR)
    test_csv = pd.read_csv(TEST_DATA_DIR)

    train_set = FashionDataset(
        train_csv, transform=transforms.Compose([transforms.ToTensor()])
    )
    test_set = FashionDataset(
        test_csv, transform=transforms.Compose([transforms.ToTensor()])
    )

    train_loader = DataLoader(train_set, batch_size=100)
    test_loader = DataLoader(test_set, batch_size=100)

    model = FashionCNN()
    model.to(device)

    error = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    num_epochs = 1
    count = 0

    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    for _ in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is
            # no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

            # Testing the model

            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print(
                    "Iteration: {}, Loss: {}, Accuracy: {}%".format(
                        count, loss.data, accuracy
                    )
                )

    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.savefig("loss.png")


if __name__ == "__main__":
    main()
