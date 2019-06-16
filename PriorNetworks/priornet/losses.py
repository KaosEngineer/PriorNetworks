import torch
import torch.nn.functional as F


class PriorNetLoss(object):
    """
    Can be applied to ANY model which returns LOGITS

    """

    def __init__(self, target_concentration=1e3, smoothing=1.0, reverse_KL=True):
        self.target_concentration = torch.tensor(target_concentration, dtype=torch.float32)
        self.smoothing = smoothing
        self.reverse_KL = reverse_KL

    def __call__(self, logits, labels):
        alphas = torch.exp(logits)
        return self.forward(alphas, labels)

    def forward(self, alphas, labels):
        loss = self.compute_loss(alphas, labels)
        return torch.mean(loss)

    def compute_loss(self, alphas, labels):
        # TODO: Need to make sure this actually works right...  so that concentration is either fixed, or on a per-example setup
        target_alphas = torch.ones_like(alphas) * self.smoothing
        if labels is not None:
            target_alphas += torch.zeros_like(alphas).scatter_(1, labels[:, None], self.target_concentration)

        if self.reverse_KL:
            in_domain_loss = reverse_kl_divergence_dirichlet(alphas, target_alphas=target_alphas)
        else:
            in_domain_loss = kl_divergence_dirichlet(alphas, target_alphas=target_alphas)
        return in_domain_loss


def kl_divergence_dirichlet(alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8):
    """
    This function computes the Forward KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of forward KL divergences between target Dirichlet and model
    """
    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    alphas_term = torch.sum(torch.lgamma(alphas + epsilon) - torch.lgamma(target_alphas + epsilon)
                            + (target_alphas - alphas) * (torch.digamma(target_alphas + epsilon)
                                                          - torch.digamma(target_precision + epsilon)), dim=1)
    cost = precision_term + alphas_term
    return cost


def reverse_kl_divergence_dirichlet(alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8):
    """
    This function computes the Reverse KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of reverse KL divergences between target Dirichlet and model
    """
    return kl_divergence_dirichlet(alphas=target_alphas, target_alphas=alphas, precision=target_precision,
                                   target_precision=precision, epsilon=epsilon)


class EnDLoss(object):
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


# class EnDDSamplesLoss(object):
#     """Standard Negative Log-likelihood of the ensemble predictions"""
#     def __init__(self, smoothing=0., teacher_prob_smoothing=1e-7):
#         self.smooth_val = smoothing
#         self.tp_scaling = 1 - teacher_prob_smoothing
#
#     def __call__(self, *args):
#         return self.forward(*args)
#
#     def forward(self, logits, teacher_logits):
#         alphas = torch.exp(logits)
#         precision = torch.sum(alphas, dim=1)
#
#         teacher_probs = F.softmax(teacher_logits, dim=1)
#         # Smooth for num. stability:
#         probs_mean = 1 / (teacher_probs.size()[1])
#         # Subtract mean, scale down, add mean back)
#         teacher_probs = self.tp_scaling * (teacher_probs - probs_mean) + probs_mean
#
#         assert torch.all(teacher_probs != 0).item()
#         log_teacher_probs_geo_mean = torch.mean(torch.log(teacher_probs), dim=2)
#
#         # Define the cost in two parts (dependent on targets and independent of targets)
#         target_independent_term = torch.sum(torch.lgamma(alphas + self.smooth_val), dim=1) - torch.lgamma(
#             precision + self.smooth_val)
#         target_dependent_term = - torch.sum((alphas - 1.) * log_teacher_probs_geo_mean, dim=1)
#         cost = target_dependent_term + target_independent_term
#
#         assert torch.all(torch.isfinite(log_teacher_probs_geo_mean)).item()
#         assert torch.all(torch.isfinite(cost)).item()
#
#         return torch.mean(cost)
#
# class EnDDSamplesSmoothLoss(object):
#     """KL divergence between output parametrised Dirchlet and a Dirichlet centered on each of ensemble's predictions
#     with a fixed precision."""
#     def __init__(self, teacher_precision=1e3, smoothing=0.):
#         self.smooth_val = smoothing
#         self.teacher_prec = torch.tensor(teacher_precision, dtype=torch.float32)
#
#     def __call__(self, *args):
#         return self.forward(*args)
#
#     def forward(self, logits, teacher_logits):
#         alphas = torch.exp(logits)
#         precision = torch.sum(alphas, dim=1)
#         alphas = alphas[:, :, None]
#         print(alphas)
#
#         teacher_probs = F.softmax(teacher_logits, dim=1)
#         teacher_alphas = teacher_probs * self.teacher_prec
#
#         prec_term1 = torch.lgamma(self.teacher_prec)
#         prec_term2 = - torch.lgamma(precision)
#         # print('prec:\n', precision, '\nprecterm 2:\n', prec_term2)
#         lgamma_term = torch.sum(torch.lgamma(alphas) - torch.lgamma(teacher_alphas) \
#                                 + (teacher_alphas - alphas) * \
#                                 (torch.digamma(teacher_alphas) - torch.digamma(self.teacher_prec)), dim=1)
#         print(lgamma_term)
#         cost = torch.sum(prec_term1[:, None] + prec_term2[:, None] + lgamma_term, dim=1)
#
#         assert torch.all(torch.isfinite(cost)).item()
#
#         return torch.mean(cost)
