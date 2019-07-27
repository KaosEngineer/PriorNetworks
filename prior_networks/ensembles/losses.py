import torch
from torch.nn import functional as F


class EnDLoss:
    def __init__(self, temp=1.0):
        self.temp = temp  # Distillation temperature (scaling of logits to emphasise differences within ensemble)

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, logits, teacher_logits):
        """

        :param logits:
        :param teacher_logits:
        :return:
        """
        teacher_probs = F.softmax(teacher_logits / self.temp, dim=1)
        teacher_probs_mean = torch.mean(teacher_probs, dim=2)

        cost = - teacher_probs_mean * F.log_softmax(logits / self.temp, dim=1)

        assert torch.all(torch.isfinite(cost)).item()

        return torch.mean(cost)
