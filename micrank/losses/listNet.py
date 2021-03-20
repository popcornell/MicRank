import torch
import torch.nn.functional as F
from itertools import permutations

PADDED_Y_VALUE = -1e30
DEFAULT_EPS=1e-8


def sample_good_perms(y_true, l):

    # we sort y_pred get top k
    #y_true = y_true.detach().cpu().numpy()
    y_true = [(i, x) for i, x in enumerate(y_true)]
    y_true = sorted(y_true, key=lambda x: x[-1], reverse=True)
    top_k = y_true[:sum([x[-1] == max(y_true)[-1] for x in y_true])]
    last_k_indexes = y_true[sum([x == max(y_true) for x in y_true]):]
    last_k_indexes = [x[0] for x in last_k_indexes]

    out = []
    perms = iter(permutations([x[0] for x in top_k]))
    for i in range(l):
        try:
            out.append(list(next(perms)) + last_k_indexes)
        except StopIteration:
            break

    return out


def AdaptiveTopKListNet(y_pred, y_true, l=4):

    assert y_pred.size(0) == 1, "support batch size 1 for now"

    perms = sample_good_perms(y_true[0].detach().cpu().numpy(), l)

    losses = []
    for p in perms:
        # permuting y_pred and y_true
        c_true = y_true[0, p]
        c_pred = y_pred[0, p]

        preds_smax = F.softmax(c_pred)
        true_smax = F.softmax(c_true)

        preds_smax = preds_smax + 1e-8
        preds_log = torch.log(preds_smax)

        losses.append(-torch.sum(true_smax * preds_log))

    tot_losses = torch.sum(torch.stack(losses))

    return tot_losses


def listNet(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    preds_smax = F.softmax(y_pred, dim=1)
    true_smax = F.softmax(y_true, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
