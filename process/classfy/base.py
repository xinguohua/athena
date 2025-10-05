from abc import ABC, abstractmethod
import torch

class BaseClassify(ABC):
    def __init__(self):
        self.model = None


    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _train_loop(self, embeddings, **kwargs):
        pass

    def train(self, embeddings, **kwargs):
        self.model = self._build_model()
        self._train_loop(embeddings, **kwargs)
        return self

    def predict(self, embeddings):
        assert self.model is not None, "model 未训练"
        self.model.eval()
        with torch.no_grad():
            return self.model(embeddings)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        pass