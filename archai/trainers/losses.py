import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.optim import SGD, Adam, lr_scheduler


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            # For label smoothing, we replace 1-hot vector with 0.9-hot vector instead.
            # Create empty vector of same size as targets, fill them up with smoothing/(n-1)
            # then replace element where 1 supposed to go and put there 1-smoothing instead
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:  # to support weighted targets
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# Credits: https://github.com/NVIDIA/DeepLearningExamples/blob/342d2e7649b9a478f35ea45a069a4c7e6b1497b8/PyTorch/Classification/ConvNets/main.py#L350
class NLLMultiLabelSmooth(nn.Module):
    """According to NVidia code, this should be used with mixup?"""

    def __init__(self, smoothing=0.0):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
