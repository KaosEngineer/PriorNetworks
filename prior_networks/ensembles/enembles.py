import torch
import torch.functional as F
from prior_networks.util_pytorch import get_grid, load_model, save_model
from pathlib import Path

class Ensemble(object):
    def __init__(self, models):
        assert type(models) is list
        self.models = models

    def __call__(self, x):
        return self.forward(x)

    def __len__(self):
        return len(self.models)

    def forward(self, x):
        y = []
        for model in self.models:
            y.append(model(x))
        y = torch.stack(y, dim=2)
        return y

    def avg_predict(self, x):
        return torch.mean(self.forward(x), dim=2)

    def predict(self, x):
        return torch.mean(F.softmax(self.forward(x), dim=1), dim=2)

    @classmethod
    def load_from_savefile(cls, savedir_path, model_class, n_models=None):
        """Construct the ensemble object from a directory with several model savefiles"""
        models = []

        savedir_path = Path(savedir_path)
        save_files = [f for f in savedir_path.iterdir() if f.is_file() and f.suffix == '.pt']
        if n_models is not None:
            assert len(save_files) >= n_models
            save_files = save_files[:n_models]
        for save_file in save_files:
            models.append(load_model(model_class, save_file))
        return cls(models)

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()

    def to(self, device):
        for model in self.models:
            model.to(device)

    def eval_probs_on_grid(self, extent, res=400):
        """
        Evaluate the ensemble on a res x res grid spanning from [-extent, extent].

        :return: Numpy array of probabilities predicted by the model with
        shape [num_eval_points, num_classes, num_models]
        """
        xrange = (-extent, extent)
        yrange = (-extent, extent)

        xx, yy = get_grid(xrange, yrange, res)
        eval_points = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=1))
        with torch.no_grad():
            probs = F.softmax(self(eval_points), dim=1).cpu().numpy()
        return probs
