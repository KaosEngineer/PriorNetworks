import context
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple


def eval_logits_on_dataset(model: nn.Module, dataset: Dataset, batch_size: int = 128,
                           device: Optional[torch.device] = None,
                           num_workers: int = 4) -> Tuple[torch.tensor, torch.tensor]:
    """
    Takes a model and an evaluation dataset, and returns the logits
    output by the model on that dataset as an array
    :param model: torch.nn.Module that outputs model logits
    :param dataset: pytorch dataset with inputs and labels
    :param batch_size: int
    :param device: device to use for evaluation
    :param num_workers: int, num. workers for the data loader
    :return: stacked torch tensor of logits returned by the model
    on that dataset, and the labels
    """
    # Set model in eval mode
    model.eval()

    testloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, labels = data
            if device is not None:
                inputs, labels = map(lambda x: x.to(device),
                                     (inputs, labels))
                logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits.cpu(), labels.cpu()

