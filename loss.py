import torch
from torch.distributions import Categorical
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cross_entropy_loss averaged: see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean").to(device)

soft_max = nn.Softmax(dim=1)


def margin_loss(outputs, target, beta=0.5, m=0.4):
    id_outputs, ood_outputs = outputs

    ce_loss = 0
    for j in range(len(id_outputs)):
        ce_loss += cross_entropy_loss(id_outputs[j], target[j])

    ce_loss /= len(id_outputs)

    id_entropy = 0
    for j in range(len(id_outputs)):
        sm = soft_max(id_outputs[j])
        entropy = Categorical(probs=sm).entropy()
        id_entropy += (torch.sum(entropy) / entropy.size()[0])

    id_entropy /= len(id_outputs)

    ood_entropy = 0
    for j in range(len(ood_outputs)):
        sm = soft_max(ood_outputs[j])
        entropy = Categorical(probs=sm).entropy()
        ood_entropy += (torch.sum(entropy) / entropy.size()[0])

    ood_entropy /= len(ood_outputs)

    m_loss = m + id_entropy - ood_entropy

    if m_loss > 0:
        return ce_loss + beta * m_loss

    return ce_loss


