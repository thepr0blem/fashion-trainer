import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fashion_trainer import plot
from fashion_trainer.constants import IMAGE_SIZE, NUMBER_OF_CLASSES
from fashion_trainer.dataset import FashionDataset, get_label_name
from fashion_trainer.model import FashionCNN
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_csv: str,
        test_csv: str,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 100,
    ):
        self.train_loader = self.get_data_loader(
            data_csv=train_csv, batch_size=batch_size
        )
        self.test_loader = self.get_data_loader(data_csv=test_csv, batch_size=batch_size)
        self.device = self.get_device()
        self.model = FashionCNN()
        self.error = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    @staticmethod
    def get_data_loader(data_csv: str, batch_size: int) -> DataLoader:
        df = pd.read_csv(data_csv)
        dataset = FashionDataset(
            data=df, transform=transforms.Compose([transforms.ToTensor()])
        )
        return DataLoader(dataset, batch_size=batch_size)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.model.to(self.device)

        count = 0

        loss_list = []
        iteration_list = []
        accuracy_list = []

        predictions_list = []
        labels_list = []

        for epoch in range(self.num_epochs):
            print(f"Epoch -- {epoch + 1} --")

            for images, labels in self.train_loader:
                # Transfering images and labels to GPU if available
                images, labels = images.to(self.device), labels.to(self.device)

                train = Variable(images.view(self.batch_size, 1, IMAGE_SIZE, IMAGE_SIZE))
                labels = Variable(labels)

                # Forward pass
                outputs = self.model(train)
                loss = self.error(outputs, labels)

                # Initializing a gradient as 0 so there is
                # no mixing of gradient among the batches
                self.optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Optimizing the parameters
                self.optimizer.step()
                count += 1

                # Testing the model
                if not count % 50:
                    total = 0
                    correct = 0

                    for images, labels in self.test_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels_list.append(labels)

                        test = Variable(images.view(100, 1, IMAGE_SIZE, IMAGE_SIZE))

                        outputs = self.model(test)

                        predictions = torch.max(outputs, 1)[1].to(self.device)
                        predictions_list.append(predictions)
                        correct += (predictions == labels).sum()

                        total += len(labels)

                    accuracy = torch.true_divide(correct * self.batch_size, total)
                    loss_list.append(loss.data)
                    iteration_list.append(count)
                    accuracy_list.append(accuracy)

                    if not count % 500:
                        print(
                            f"Iteration: {count}, Loss: {loss.data:.2f}, "
                            f"Accuracy: {accuracy:.2f}%"
                        )

        plot.plot_loss(iterations=iteration_list, losses=loss_list)
        plot.plot_accuracy(iterations=iteration_list, accuracies=accuracy_list)

        class_correct = NUMBER_OF_CLASSES * [0.0]
        total_correct = NUMBER_OF_CLASSES * [0.0]

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                test = Variable(images)
                outputs = self.model(test)
                predicted = torch.max(outputs, 1)[1]
                c = (predicted == labels).squeeze()

                for i in range(100):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    total_correct[label] += 1

        print("== Accuracy per class: ==")
        for i in range(NUMBER_OF_CLASSES):
            print(
                f"Accuracy of {get_label_name(i)}: "
                f"{class_correct[i] * 100 / total_correct[i]:.2f}%"
            )

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)
