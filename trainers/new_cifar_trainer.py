from trainers.abc_trainer import abcTrainer
import torch
from ood_metrics import fpr_at_95_tpr

class Cifar10Trainer(abcTrainer):
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        super().__init__(dataloader, net, loss, optimizer, device, validation_frequency=validation_frequency, max_epoch=max_epoch)
  
        
    def train_epoch(self):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(self.dataloader, 0):
            prev_loss = 999999
            curr_loss = 0
            #TODO: insert a convergent statement: e.g. 
            if prev_loss > curr_loss:
                prev_loss = curr_loss
                
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
                curr_loss = self.loss((id_outputs, ood_outputs), id_labels)
                curr_loss.backward()
                self.optimizer.step()
    
                running_loss += curr_loss.item()
                
                #get accuracy
                #TODO:fix the accuracy
                id_outputs_t = torch.stack(id_outputs)
                id_labels_t = torch.stack(id_labels)
                   
                _, pred = torch.max(id_outputs_t, 2)
                total += id_labels_t.size(1)
                correct += pred.eq(id_labels_t).sum().item()
                accuracy = correct/total
                
                #validate
                #ood_score = self.validation()
                
                #TODO: compare ood score and accuracy between best model and curr moodel
                #TODO: update best model
            
            
               
        print(accuracy)
        print(f"epoch {self.epoch}: {running_loss}")
        
    def validation(self):
        #TODO: need to validate net on TinyImageNet here?
        
        #for fpr_at_95_tpr: pip install ood-metrics (temporary, we might need to implement all metrics)
        #TODO: need labels of tinyImageNet to compute ood_score
        ood_score = fpr_at_95_tpr(out, labels)
        print("Ok validation")
        return ood_score
        

    def validate(self):
        print("Ok val")
        
    