from trainers.abc_trainer import abcTrainer
import torch

class ToyTrainer(abcTrainer):
    def __init__(
        self,
        dataloader,
        net,
        loss,
        optimizer,
        device,
        validation_frequency=1,
        max_epoch=100,
    ):
        super().__init__(
            dataloader,
            net,
            loss,
            optimizer,
            device,
            validation_frequency=validation_frequency,
            max_epoch=max_epoch,
        )
        self.train_dataloader, self.validation_dataloder = dataloader

    def train_epoch(self):
        self.net.train()
        running_loss = 0.0
        running_accuracy = 0.0
        image_counter = 0
        for i, data in enumerate(self.train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == labels).item()
            running_accuracy += acc

            image_counter += labels.size()[0]
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (self.epoch + 1, i + 1, running_loss / image_counter)
                )
                running_loss = 0.0

                print(
                    "[%d, %5d] accuracy: %.3f"
                    % (self.epoch + 1, i + 1, running_accuracy / image_counter)
                )
                running_accuracy = 0.0

                image_counter = 0

    def validate(self):
        self.net.eval()
        running_accuracy = 0.0
        image_counter = 0
        for i, data in enumerate(self.validation_dataloder, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward + backward + optimize
            outputs = self.net(inputs)

            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == labels).item()
            running_accuracy += acc

            image_counter += labels.size()[0]

            # print statistics
            if i % 100 == 99:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] accuracy: %.3f"
                    % (self.epoch + 1, i + 1, running_accuracy / image_counter)
                )
                running_accuracy = 0.0

                image_counter = 0
