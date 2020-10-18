import torch
from trainers.abc_trainer import abcTrainer


class Cifar10Trainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch)
        self.train_loader, self.validation_loader = dataloader

    def train_epoch(self):
        self.net.train()
        running_loss = 0.0
        running_accuracy = 0.0
        id_images_counter = 0
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
                running_accuracy += acc
                id_images_counter += pred.size()[0]

            for j in range(ood_images.size()[1]):
                outputs = self.net(ood_images[:, j, :, :, :])
                ood_outputs.append(outputs)

            self.optimizer.zero_grad()
            loss = self.loss((id_outputs, ood_outputs), id_labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        running_accuracy /= id_images_counter

        print(f"epoch {self.epoch}: train_margin_loss: {running_loss} | train_accuracy (ids): {running_accuracy}")

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
            
            #get accuracy of model
            for j in range(len(id_labels)):
                outputs = self.net(id_images[:, j, :, :, :])
                id_labels[j] = id_labels[j].to(self.device)
                pred = torch.argmax(outputs, dim=1)
                acc = torch.sum(pred == id_labels[j]).item()
                running_accuracy += acc
                id_images_counter += pred.size()[0]
            
            """
            #get OOD accuracy of model
            for j in range(len(ood_labels)):
                #TODO: get outputs_ood from algorithm 2: OOD score
                #ood_score = algorithm_2(ood_images[:, j, :, :, :])
                
                #ood_score low for ood and high or id
                threshold = 0.3
                acc_ood = torch.sum(ood_score < threshold).item()
                running_accuracy_ood += acc_ood
                ood_images_counter += pred_ood.size()[0]
        
        
        running_accuracy /= id_images_counter
        print(f"epoch {self.epoch}: OOD_accuracy (oods): {running_accuracy}")
        """
        running_accuracy /= id_images_counter
        print(f"epoch {self.epoch}: validation_accuracy (ids): {running_accuracy}")
        
        #save the best model
        