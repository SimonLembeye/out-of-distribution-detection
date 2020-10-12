
class Trainer:
    def __init__(self, dataloader, net, loss, optimizer, device, validation_frequency=1, max_epoch=100):
        self.dataloader = dataloader
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.epoch = 0
        self.validation_frequency = validation_frequency
        self.max_epoch = max_epoch
        self.device = device
        self.save_dir = "best_model_weight"

    def train(self):
        for epoch in range(0, self.max_epoch):
            self.epoch = epoch

            # training
            self.train_epoch()

            # validation
            if epoch % self.validation_frequency == 0:
                self.validate()

        print(f'Train finished')


    def train_epoch(self):
        running_loss = 0.0
        for i, data in enumerate(self.dataloader, 0):
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

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (self.epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    def validate(self):
        print("Ok val")