from foolbox.batch_attacks import EADAttack
import numpy as np

class AdaptiveEADAttack(EADAttack):
    """Gradient based attack which uses an elastic-net regularization [1].
    This implementation is based on the attacks description [1] and its
    reference implementation [2].

    References
    ----------
    .. [1] Pin-Yu Chen (*), Yash Sharma (*), Huan Zhang, Jinfeng Yi,
           Cho-Jui Hsieh, "EAD: Elastic-Net Attacks to Deep Neural
           Networks via Adversarial Examples",
           https://arxiv.org/abs/1709.04114

    .. [2] Pin-Yu Chen (*), Yash Sharma (*), Huan Zhang, Jinfeng Yi,
           Cho-Jui Hsieh, "Reference Implementation of 'EAD: Elastic-Net
           Attacks to Deep Neural Networks via Adversarial Examples'",
           https://github.com/ysharma1126/EAD_Attack/blob/master/en_attack.py
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
        is_adv_loss_grad = yield from a.backward_one(logits_diff_grad, x)
        assert is_adv_loss >= 0
        if is_adv_loss == 0:
            is_adv_loss_grad = 0

        squared_l2_distance_grad = (2 / s**2) * (x - reconstructed_original)

        total_loss_grad = squared_l2_distance_grad + const * is_adv_loss_grad
        return total_loss, total_loss_grad