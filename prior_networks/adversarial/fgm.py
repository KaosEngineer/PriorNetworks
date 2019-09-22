import torch
from torch import nn


def construct_fgm_attack(model,
                         inputs,
                         labels,
                         epsilon,
                         criterion=nn.CrossEntropyLoss(),
                         device=None,
                         pin_memory: bool = True):
    adv_inputs = inputs.clone()
    adv_inputs.requires_grad = True
    model.eval()

    with torch.enable_grad():
        outputs = model(adv_inputs)

        epsilon = epsilon.view([outputs.size()[0], 1, 1, 1])

        if device is not None:
            epsilon = epsilon.to(device, non_blocking=pin_memory)

        loss = criterion(outputs, labels)
        assert torch.all(torch.isfinite(loss)).item()

        grad_outputs = torch.ones(loss.shape)
        if device is not None:
            grad_outputs = grad_outputs.to(device, non_blocking=pin_memory)

        grads = torch.autograd.grad(loss,
                                    adv_inputs,
                                    grad_outputs=grad_outputs,
                                    only_inputs=True)[0]

        update = epsilon * grads.sign()

        perturbed_image = adv_inputs + update
        adv_inputs.data = perturbed_image

    return adv_inputs
