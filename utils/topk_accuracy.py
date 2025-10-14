import torch

def topk_accuracy(outputs, labels, k=5):
    _, pred_topk = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk))
    return correct.any(dim=1).float().mean().item()
