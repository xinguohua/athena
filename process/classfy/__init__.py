from .base import BaseClassify
from .prographer_classify import PrographerClassify, PrographerConfig
from .unicorn_classify import UnicornClassify, UnicornConfig
from .svm_classify import TopKDeviationClassify, TopKDeviationConfig
from .mlp_classify import MLPClassify, MLPConfig
# 兼容旧名：SVMClassify/SVMConfig 指向 Top-K 偏离度实现
SVMClassify = TopKDeviationClassify
SVMConfig = TopKDeviationConfig

__all__ = [
    "BaseClassify",
    "PrographerClassify",
    "UnicornClassify",
    "TopKDeviationClassify",
    "TopKDeviationConfig",
    "MLPClassify",
    "MLPConfig",
    # 兼容旧名
    "SVMClassify",
    "SVMConfig",
]

def get_classfy(name: str, **kwargs) -> BaseClassify:
    """
    工厂函数，根据名字返回对应的 Trainer 实例
    """
    trainers = {
        "prographer": PrographerClassify,
        "unicorn": UnicornClassify,
        # 新增显式名称
        "topk": TopKDeviationClassify,
        "topk_deviation": TopKDeviationClassify,
        # 论文方法：两层 MLP + cross-entropy
        "mlp": MLPClassify,
    }
    if name not in trainers:
        raise ValueError(f"未知训练器: {name}, 可选: {list(trainers.keys())}")
    return trainers[name](**kwargs)