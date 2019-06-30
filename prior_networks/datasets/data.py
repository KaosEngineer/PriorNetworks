from torch.utils.data import Dataset


class DummyLabelWrapper(Dataset):
    """
    A wrapper to wrap a dataset of only inputs into a dataset that returns inputs and dummy labels.
    Useful when concatenating an OOD dataset without labels with a training dataset with labels.
    """
    def __init__(self, dataset, dummy_label=1):
        self.dataset = dataset
        self.dummy_label = dummy_label

    def __getitem__(self, idx):
        return self.dataset[idx], self.dummy_label

    def __len__(self):
        return len(self.dataset)
