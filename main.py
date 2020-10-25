import torch
import os
import operator

from torch.autograd import Variable

from datasets.cifar10 import Cifar10Dataset
from datasets.tiny_imagenet import TinyImagenetDataset
from loss import margin_loss
from models.dense_net import DenseNet
from models.toy_net import ToyNet
import torch.optim as optim

from models.wide_res_net import WideResNet
from models.wideresnet import WideResNetFb
from trainers.cifar_trainer import Cifar10Trainer
from torch.distributions import Categorical
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = nn.Softmax(dim=0)

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# NET = ToyNet(class_nb=8).to(device)
# NET = DenseNet(num_classes=8, depth=50).to(device)
# NET = WideResNet(8).to(device)
NET = WideResNetFb(8).to(device)


class Classifier:
    def __init__(self, train_name="toy_train", id=0, class_to_id={}):
        ood_list = []
        id_list = []
        self.id_to_class = {}
        for class_name in CLASSES:
            if class_name in class_to_id.keys():
                id_list.append(class_name)
                self.id_to_class[class_to_id[class_name]] = class_name
            else:
                ood_list.append(class_name)

        dir_path = os.path.join("train_models_weights", train_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.id = id
        self.train_name = train_name
        self.best_weights_path = os.path.join(
            "train_models_weights", train_name, f"{id}.pth"
        )

        self.train_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "train"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=20, shuffle=True, num_workers=3
        )

        self.validation_dataset = Cifar10Dataset(
            data_dir=os.path.join("data", "cifar-10", "test"),
            id_class_list=id_list,
            ood_class_list=ood_list,
            class_to_id=class_to_id,
        )

        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=10, shuffle=True, num_workers=3
        )

    def get_and_update_current_trainer(self):

        if os.path.exists(self.best_weights_path):
            NET.load_state_dict(torch.load(self.best_weights_path))

        NET.train()

        optimizer = optim.SGD(NET.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        trainer = Cifar10Trainer(
            dataloader=[self.train_loader, self.validation_loader],
            net=NET,
            loss=margin_loss,
            optimizer=optimizer,
            device=device,
            max_epoch=1,
        )
        self.trainer = trainer
        return trainer


def validation(classifiers, dataset, w_label=False):

    batch_size = 25
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    image_counter = 0

    temperature = 100
    epsilon = 0.002

    scores_0 = {
        "airplane": 0,
        "automobile": 0,
        "bird": 0,
        "cat": 0,
        "deer": 0,
        "dog": 0,
        "frog": 0,
        "horse": 0,
        "ship": 0,
        "truck": 0
    }

    classifiers_scores_list = [[] for _ in range(len(classifiers))]
    classifiers_ood_scores_list = [[] for _ in range(len(classifiers))]
    labels_list = []

    for j in range(len(classifiers)):

        clf = classifiers[j]
        if os.path.exists(clf.best_weights_path):
            NET.load_state_dict(torch.load(clf.best_weights_path))

        NET.eval()

        image_counter = 0

        scores_list = []
        ood_scores_list = []


        for i, data in enumerate(loader, 0):

            if i > 24:
                break

            images, labels = data
            images = Variable(images.to(device), requires_grad=True)

            NET.zero_grad()
            out = NET(images)

            # Prediction and entropy with temperature scaling
            f_x = soft_max(out / temperature)
            entropy = Categorical(probs=f_x).entropy()

            # Compute gradient over entropy loss step
            loss = entropy.sum()
            loss.backward(retain_graph=True)
            x_ = images - epsilon * torch.sign(images.grad)

            # Compute OOD scores
            out_ = NET(x_)
            f_x_ = soft_max(out_ / temperature)
            entropy_ = Categorical(probs=f_x).entropy()

            ood_scores = (torch.max(f_x_) - entropy_)

            for k in range(len(out)):
                res = out[k]
                img_scores = scores_0.copy()
                for p in range(len(res)):
                    img_scores[clf.id_to_class[p]] += res[p].item()

                scores_list.append(img_scores)
                ood_scores_list.append(ood_scores[k].item())
                image_counter += 1

                if j == 0 and w_label:
                    labels_list.append(labels.detach().cpu().numpy()[k])

        classifiers_scores_list[j] = scores_list
        classifiers_ood_scores_list[j] = ood_scores_list

    images_nb = len(classifiers_ood_scores_list[0])
    ood_scores_final_list = [0 for _ in range(images_nb)]
    prediction_final_list = ["" for _ in range(images_nb)]

    for i in range(images_nb):

        running_score = scores_0.copy()

        for j in range(len(classifiers)):
            ood_scores_final_list[i] += classifiers_ood_scores_list[j][i]

            for c in scores_0.keys():
                running_score[c] += classifiers_scores_list[j][i][c]

        prediction_final_list[i] = max(running_score.items(), key=operator.itemgetter(1))[0]

    print("sum ood", sum(ood_scores_final_list))
    print(prediction_final_list)
    print(ood_scores_final_list)

    if w_label:
        id_to_class = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }
        acc = 0
        for m in range(images_nb):
            if id_to_class[labels_list[m]] == prediction_final_list[m]:
                acc += 1

        print("Accuracy: ", acc / images_nb)


if __name__ == "__main__":
    class_to_id_list = [
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "airplane": 0,
            "automobile": 1,
            "deer": 2,
            "dog": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
        {
            "bird": 0,
            "cat": 1,
            "deer": 2,
            "dog": 3,
            "frog": 4,
            "horse": 5,
            "ship": 6,
            "truck": 7,
        },
    ]

    classifiers = [
        Classifier(
            class_to_id=class_to_id_list[k], train_name="wide_fb_train_102502", id=k
        )
        for k in range(len(class_to_id_list))
    ]

    for _ in range(200):
        print()
        for classifier in classifiers:
            print()
            print(f"## Train classifier {classifier.id}!")
            trainer = classifier.get_and_update_current_trainer()

            trainer.train()
            torch.save(trainer.net.state_dict(), classifier.best_weights_path)

        print()
        print("Validation CIFAR10")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        cifar_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        validation(classifiers, cifar_dataset, w_label=True)

        print()
        print("Validation Tinyimagenet")
        tiny_dataset = TinyImagenetDataset(
            data_dir=os.path.join("data", "tiny-imagenet-200", "val", "images"),
        )
        validation(classifiers, tiny_dataset)










