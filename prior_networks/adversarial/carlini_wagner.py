from foolbox.attacks import CarliniWagnerL2Attack
import numpy as np

class AdaptiveCarliniWagnerL2Attack(CarliniWagnerL2Attack):
    """The Adaptive L2 version of the Carlini & Wagner attack.
    This attack is described in [1]_. This implementation
    is based on the reference implementation by Carlini [2]_.
    For bounds ≠ (0, 1), it differs from [2]_ because we
    normalize the squared L2 loss with the bounds.
    References
    ----------
    .. [1] Nicholas Carlini, David Wagner: "Towards Evaluating the
           Robustness of Neural Networks", https://arxiv.org/abs/1608.04644
    .. [2] https://github.com/carlini/nn_robust_attacks
    """

    @classmethod
    def loss_function(cls, const, a, x, logits, reconstructed_original,
                      confidence, min_, max_):
        """Returns the loss and the gradient of the loss w.r.t. x,
        assuming that logits = model(x)."""

        targeted = a.target_class() is not None
        if targeted:
            c_minimize = cls.best_other_class(logits, a.target_class())
            c_maximize = a.target_class()
        else:
            c_minimize = a.original_class
            c_maximize = cls.best_other_class(logits, a.original_class)

        is_adv_loss = logits[c_minimize] - logits[c_maximize]

        # is_adv is True as soon as the is_adv_loss goes below 0
        # but sometimes we want additional confidence
        is_adv_loss += confidence
        is_adv_loss = max(0, is_adv_loss)

        s = max_ - min_
        squared_l2_distance = np.sum((x - reconstructed_original)**2) / s**2
        total_loss = squared_l2_distance + const * is_adv_loss

        # calculate the gradient of total_loss w.r.t. x
        logits_diff_grad = np.zeros_like(logits)
        logits_diff_grad[c_minimize] = 1
        logits_diff_grad[c_maximize] = -1
        is_adv_loss_grad = a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s**2) * (x - reconstructed_original)

        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad