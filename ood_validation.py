import operator
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from classifier import Classifier
from metrics import get_metrics
from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

soft_max = torch.nn.Softmax(dim=0)


def compute_ood_scores(
    classifiers,
    dataset,
    temperature=1,
    epsilon=0,
    batch_size=25,
    num_epoch=25,
    w_label=False,
):

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

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
        "truck": 0,
    }

    classifiers_scores_list = [[] for _ in range(len(classifiers))]
    classifiers_ood_scores_list = [[] for _ in range(len(classifiers))]
    labels_list = []

    for j in range(len(classifiers)):

        clf = classifiers[j]
        if os.path.exists(clf.best_weights_path):
            clf.net.load_state_dict(torch.load(clf.best_weights_path))

        clf.net.eval()

        image_counter = 0

        scores_list = []
        ood_scores_list = []

        for i, data in enumerate(loader, 0):

            if i > num_epoch - 1:
                break

            images, labels = data
            images = Variable(images.to(device), requires_grad=True)

            clf.net.zero_grad()
            out = clf.net(images)

            # Prediction and entropy with temperature scaling
            f_x = soft_max(out / temperature)
            entropy = Categorical(probs=f_x).entropy()

            # Compute gradient over entropy loss step
            loss = entropy.sum()
            loss.backward(retain_graph=True)
            x_ = images - epsilon * torch.sign(images.grad)

            # Compute OOD scores
            out_ = clf.net(x_)
            f_x_ = soft_max(out_ / temperature)
            entropy_ = Categorical(probs=f_x).entropy()

            ood_scores = torch.max(f_x_) - entropy_

            sm_out = soft_max(out)
            for k in range(len(out)):
                res = sm_out[k]
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

        prediction_final_list[i] = max(
            running_score.items(), key=operator.itemgetter(1)
        )[0]

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

    return ood_scores_final_list


def get_validation_metrics(
    id_dataset,
    ood_dataset,
    net_architecture="ToyNet",
    train_name="toy_train_102501",
    class_to_id_list=[],
    temperature=1,
    epsilon=0,
    batch_size=25,
    num_epoch=25,
):
    if net_architecture == "DenseNet":
        net = DenseNet(num_classes=8, depth=50).to(device)
    elif net_architecture == "WideResNet":
        net = WideResNet(8).to(device)
    else:
        net = ToyNet(class_nb=8).to(device)

    print(ood_dataset.name)

    classifiers = [
        Classifier(net, class_to_id=class_to_id_list[k], train_name=train_name, id=k)
        for k in range(len(class_to_id_list))
    ]

    ood_scores_id_data = np.array(
        compute_ood_scores(
            classifiers,
            id_dataset,
            temperature=temperature,
            epsilon=epsilon,
            batch_size=batch_size,
            num_epoch=num_epoch,
            w_label=True,
        )
    )
    ood_scores_ood_data = np.array(
        compute_ood_scores(
            classifiers,
            ood_dataset,
            temperature=temperature,
            epsilon=epsilon,
            batch_size=batch_size,
            num_epoch=num_epoch,
        )
    )

    plt.figure()
    bins = np.linspace(
        min(np.min(ood_scores_id_data), np.min(ood_scores_ood_data)),
        max(np.max(ood_scores_id_data), np.max(ood_scores_ood_data)),
        100,
    )
    plt.hist(ood_scores_id_data, bins, alpha=0.5, label="id")
    plt.hist(ood_scores_ood_data, bins, alpha=0.5, label="ood")
    plt.legend(loc="upper right")
    plt.savefig(
        os.path.join(
            "distributions",
            f"{train_name}_{ood_dataset.name}_{temperature}_{epsilon}.jpg",
        )
    )

    labels = np.concatenate(
        (np.ones_like(ood_scores_id_data), np.zeros_like(ood_scores_ood_data)), axis=0
    )
    ood_scores = np.concatenate((ood_scores_id_data, ood_scores_ood_data), axis=0)

    fpr95, auroc, aupr_in, aupr_out, error = get_metrics(labels, ood_scores)

    print("detection_error", error)
    print("fpr95", fpr95)
    print("auroc", auroc)
    print("aupr_in", aupr_in)
    print("aupr_out", aupr_out)
