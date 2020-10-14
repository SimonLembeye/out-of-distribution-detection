from trainers.abc_trainer import abcTrainer


class Cifar10Trainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch)

    def train_epoch(self):
        running_loss = 0.0
        for i, data in enumerate(self.dataloader, 0):
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

            for j in range(ood_images.size()[1]):
                outputs = self.net(ood_images[:, j, :, :, :])
                ood_outputs.append(outputs)

            self.optimizer.zero_grad()
            loss = self.loss((id_outputs, ood_outputs), id_labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(running_loss)

    def validate(self):
        print("Ok val")