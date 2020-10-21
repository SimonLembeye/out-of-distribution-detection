import torch
from trainers.abc_trainer import abcTrainer
import time


class Cifar10Trainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch)
        self.train_loader, self.validation_loader = dataloader

    def train_epoch(self):
        self.net.train()
        start = time.time()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_id_images_counter = 0

        running_loss = 0.0
        running_accuracy = 0.0
        running_id_images_counter = 0

        for i, data in enumerate(self.train_loader, 0):
            id_images, ood_images, id_labels, ood_labels = (
                data["id_images"],
                data["ood_images"],
                data["id_labels"],
                data["ood_labels"],
            )

            id_outputs = []
            ood_outputs = []

            id_images = id_images.to(self.device)
            ood_images = ood_images.to(self.device)

            for j in range(len(id_labels)):
                outputs = self.net(id_images[:, j, :, :, :])
                id_outputs.append(outputs)
                id_labels[j] = id_labels[j].to(self.device)

                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == id_labels[j]).item()
                epoch_accuracy += acc
                running_accuracy += acc
                epoch_id_images_counter += pred.size()[0]
                running_id_images_counter += pred.size()[0]

            for j in range(ood_images.size()[1]):
                outputs = self.net(ood_images[:, j, :, :, :])
                ood_outputs.append(outputs)

            self.optimizer.zero_grad()
            loss = self.loss((id_outputs, ood_outputs), id_labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            running_loss += loss.item()

            if i % 5 == 0:
                print(f"step: {i} | running loss: {running_loss / running_id_images_counter} | running accuracy: {running_accuracy / running_id_images_counter} | training time: {time.time() - start}")
                running_loss = 0.0
                running_accuracy = 0.0
                running_id_images_counter = 0

        epoch_accuracy /= epoch_id_images_counter

        print(f"epoch {self.epoch}: train_margin_loss: {epoch_loss} | train_accuracy (ids): {epoch_accuracy}")

    def validate(self):
        self.net.eval()
        running_accuracy = 0.0
        id_images_counter = 0
        for i, data in enumerate(self.validation_loader, 0):
            id_images, ood_images, id_labels, ood_labels = (
                data["id_images"],
                data["ood_images"],
                data["id_labels"],
                data["ood_labels"],
            )
            id_images = id_images.to(self.device)

            for j in range(len(id_labels)):
                outputs = self.net(id_images[:, j, :, :, :])
                id_labels[j] = id_labels[j].to(self.device)
                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == id_labels[j]).item()
                running_accuracy += acc
                id_images_counter += pred.size()[0]

        running_accuracy /= id_images_counter

        print(f"epoch {self.epoch}: validation_accuracy (ids): {running_accuracy}")