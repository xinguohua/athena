from abc import ABC, abstractmethod
from typing import Optional
import torch

class BaseClassify(ABC):
    def __init__(self, gid: Optional[str] = None):
        self.model = None
        # 身份后缀（用于持久化文件名区分不同用户/运行）
        self.gid: Optional[str] = gid


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

    # 工具：为文件名添加身份后缀（"name.ext" -> "name_<gid>.ext"）。
    # 若未提供 gid 或路径无后缀，保持原样。
    def with_gid_suffix(self, path: str) -> str:
        try:
            if not path or not isinstance(path, str) or not self.gid:
                return path
            from pathlib import Path
            p = Path(path)
            return f"{p.stem}_{self.gid}{p.suffix}"
        except Exception:
            return path