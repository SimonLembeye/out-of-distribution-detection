import operator
import os

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from models.dense_net import DenseNet
from models.toy_net import ToyNet
from models.wide_res_net import WideResNet
from models.wideresnet import WideResNetFb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using gpu: %s " % torch.cuda.is_available())

soft_max = torch.nn.Softmax(dim=0)


def validation(net, classifiers, dataset, w_label=False):

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
        "truck": 0,
    }

    classifiers_scores_list = [[] for _ in range(len(classifiers))]
    classifiers_ood_scores_list = [[] for _ in range(len(classifiers))]
    labels_list = []

    for j in range(len(classifiers)):

        clf = classifiers[j]
        if os.path.exists(clf.best_weights_path):
            net.load_state_dict(torch.load(clf.best_weights_path))

        net.eval()

        image_counter = 0

        scores_list = []
        ood_scores_list = []

        for i, data in enumerate(loader, 0):

            if i > 24:
                break

            images, labels = data
            images = Variable(images.to(device), requires_grad=True)

            net.zero_grad()
            out = net(images)

            # Prediction and entropy with temperature scaling
            f_x = soft_max(out / temperature)
            entropy = Categorical(probs=f_x).entropy()

            # Compute gradient over entropy loss step
            loss = entropy.sum()
            loss.backward(retain_graph=True)
            x_ = images - epsilon * torch.sign(images.grad)

            # Compute OOD scores
            out_ = net(x_)
            f_x_ = soft_max(out_ / temperature)
            entropy_ = Categorical(probs=f_x).entropy()

            ood_scores = torch.max(f_x_) - entropy_

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
