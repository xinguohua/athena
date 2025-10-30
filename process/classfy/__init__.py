from .base import BaseClassify
from .prographer_classify import PrographerClassify, PrographerConfig
from .unicorn_classify import UnicornClassify, UnicornConfig
from .svm_classify import SVMClassify, SVMConfig

__all__ = [
    "BaseClassify",
    "PrographerClassify",
    "UnicornClassify",
    "SVMClassify"
]

def get_classfy(name: str, **kwargs) -> BaseClassify:
    """
    工厂函数，根据名字返回对应的 Trainer 实例
    """
    trainers = {
        "prographer": PrographerClassify,
        "unicorn": UnicornClassify,
        "svm": SVMClassify,  # SVM分类器（通过 svm_type='oneclass'/'binary' 控制模式）
    }
    if name not in trainers:
        raise ValueError(f"未知训练器: {name}, 可选: {list(trainers.keys())}")
    return trainers[name](**kwargs)