import torch
from torch.distributions import Categorical
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

# cross_entropy_loss averaged: see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean").to(device)

soft_max = nn.Softmax(dim=1)


def margin_loss(outputs, target, beta=0.1, m=3):
    id_outputs, ood_outputs = outputs

    ce_loss = 0
    for j in range(len(id_outputs)):
        ce_loss += cross_entropy_loss(id_outputs[j], target[j])

    id_entropy = 0
    for j in range(len(id_outputs)):
        sm = soft_max(id_outputs[j])
        entropy = Categorical(probs=sm).entropy()
        id_entropy += torch.sum(entropy)

    id_entropy /= len(id_outputs)

    ood_entropy = 0
    for j in range(len(ood_outputs)):
        sm = soft_max(ood_outputs[j])
        entropy = Categorical(probs=sm).entropy()
        ood_entropy += torch.sum(entropy)

    ood_entropy /= len(id_outputs)

    m_loss = m + id_entropy - ood_entropy
    print(ce_loss, m_loss)

    if m_loss > 0:
        return ce_loss + beta * m_loss
    return ce_loss


