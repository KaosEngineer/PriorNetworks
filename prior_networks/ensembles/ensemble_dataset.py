import context
import sys, os
import numpy as np


def get_ensemble_logits(ensemble_path, model, n_models, folder):
    model_dirs = [os.path.join(ensemble_path,
                               model + "{}".format(int(i))) for i in range(0, n_models)]

    logit_files = map(lambda model_dir: os.path.join(model_dir, folder + '/logits.txt'),
                      model_dirs)
    label_files = map(lambda model_dir: os.path.join(model_dir, folder + '/logits.txt'),
                      model_dirs)

    logits, labels = [], []
    for logit_path, label_path in zip(logit_files, label_files):
        labels.append(np.asarray(np.loadtxt(label_path, dtype=np.float32), dtype=np.int32))
        logits.append(np.loadtxt(logit_path, dtype=np.float32))

    labels = np.stack(labels, axis=1)
    logits = np.stack(logits, axis=1)
    print(labels.shape)
    return labels, logits


class EnsembleDataset(object):
    def __init__(self, dataset, dataset_parameters, ensemble_path, model_dirs, n_models, folder):
        # Initialize and Process the dataset we are wrapping
        self.dataset = dataset(**dataset_parameters)

        # Collect ensemble logits
        self.labels, self.logits = get_ensemble_logits(ensemble_path, model_dirs, n_models, folder)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        print(target.shape, self.labels.shape)
        assert target == self.labels[index]

        return img, target, self.logits[index]

    def __len__(self):
        return len(self.dataset)
