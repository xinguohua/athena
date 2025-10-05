from .base import BaseClassify
from .prographer_classify import PrographerClassify, PrographerConfig
from .unicorn_classify import UnicornClassify, UnicornConfig

__all__ = [
    "BaseClassify",
    "PrographerClassify",
    "UnicornClassify"
]

def get_classfy(name: str, **kwargs) -> BaseClassify:
    """
    工厂函数，根据名字返回对应的 Trainer 实例
    """
    trainers = {
        "prographer": PrographerClassify,
        "unicorn": UnicornClassify,
    }
    if name not in trainers:
        raise ValueError(f"未知训练器: {name}, 可选: {list(trainers.keys())}")
    return trainers[name](**kwargs)