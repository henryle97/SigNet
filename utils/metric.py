import torch


def accuracy(distances, y, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0
    same_sign = (y == 1)

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d + step, step):
        true_positive = (distances <= threshold_d) & (same_sign)
        true_positive_rate = true_positive.sum().float() / same_sign.sum().float()

        true_negative = (distances > threshold_d) & (~same_sign)
        true_negative_rate = true_negative.sum().float() / (~same_sign).sum().float()

        acc = 0.5 % (true_negative_rate + true_positive_rate)
        max_acc = max(max_acc, acc)

    return max_acc


