from typing import Dict, List

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
        num_epochs: int = 3,
        batch_size: int = 100,
    ):
        self.train_loader = self._get_data_loader(
            data_csv=train_csv, batch_size=batch_size
        )
        self.test_loader = self._get_data_loader(data_csv=test_csv, batch_size=batch_size)
        self.device = self._get_device()
        self.model = FashionCNN()
        self.error = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.losses: List = []
        self.metrics: Dict[str, List] = {
            "iterations": [],
            "train_acc": [],
            "test_acc": [],
        }
        self.count: int = 0

    def train(self) -> None:
        self.model.to(self.device)

        print("=== Training in progress ===")
        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)
        print("Training finished")

        self.save_training_metric_plots()
        self.print_per_class_accuracy()

    def print_per_class_accuracy(self) -> None:
        class_correct = NUMBER_OF_CLASSES * [0.0]
        total_correct = NUMBER_OF_CLASSES * [0.0]

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                test = Variable(images)
                outputs = self.model(test)
                predicted = torch.max(outputs, 1)[1]
                c = (predicted == labels).squeeze()

                for i in range(self.batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    total_correct[label] += 1

        print("=== Accuracy per class ===")
        for i in range(NUMBER_OF_CLASSES):
            print(
                f"Accuracy of {get_label_name(i)}: "
                f"{class_correct[i] * 100 / total_correct[i]:.2f}%"
            )

    def save_training_metric_plots(self) -> None:
        plot.save_loss_plot(iterations=self.metrics["iterations"], losses=self.losses)
        plot.save_accuracy_plot(
            train_accuracy=self.metrics["train_acc"],
            test_accuracy=self.metrics["test_acc"],
            iterations=self.metrics["iterations"],
        )

    def save_model(self, model_path: str) -> None:
        torch.save(self.model.state_dict(), model_path)

    def _train_epoch(self, epoch: int) -> None:
        for images, labels in self.train_loader:

            self._train_iteration(images=images, labels=labels)
            num_batches = len(self.train_loader)

            if (self.count + 1) % 100 == 0:
                self.metrics["iterations"].append(self.count)
                train_acc = self._get_accuracy(dataloader=self.train_loader)
                test_acc = self._get_accuracy(dataloader=self.test_loader)
                self.metrics["train_acc"].append(train_acc)
                self.metrics["test_acc"].append(test_acc)

                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], "
                    f"Step [{(self.count + 1) - epoch * num_batches}/{num_batches}], "
                    f"Loss: {self.losses[-1]:.4f}, "
                    f"Train Accuracy: {train_acc:.2f}, "
                    f"Test Accuracy: {test_acc:.2f}"
                )

    def _train_iteration(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.count += 1

        # Transfering images and labels to GPU if available
        images, labels = images.to(self.device), labels.to(self.device)

        train = Variable(images.view(self.batch_size, 1, IMAGE_SIZE, IMAGE_SIZE))
        labels = Variable(labels)

        # Forward pass
        outputs = self.model(train)
        loss = self.error(outputs, labels)

        # Backpropagation and optimizing the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.count + 1) % 100 == 0:
            self.losses.append(loss.data)

    @staticmethod
    def _get_data_loader(data_csv: str, batch_size: int) -> DataLoader:
        df = pd.read_csv(data_csv)
        dataset = FashionDataset(
            data=df, transform=transforms.Compose([transforms.ToTensor()])
        )
        return DataLoader(dataset, batch_size=batch_size)

    @staticmethod
    def _get_device() -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _get_accuracy(self, dataloader: DataLoader) -> torch.Tensor:

        total = len(dataloader)
        correct = 0

        predictions_list = []
        labels_list = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            labels_list.append(labels)

            test = Variable(images.view(self.batch_size, 1, IMAGE_SIZE, IMAGE_SIZE))

            outputs = self.model(test)

            predictions = torch.max(outputs, 1)[1].to(self.device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()

            total += len(labels)

        return torch.true_divide(correct * self.batch_size, total)
