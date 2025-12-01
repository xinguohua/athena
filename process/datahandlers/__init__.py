from typing import Optional
from .darpa_handler import DARPAHandler
from .darpa_handler5 import DARPAHandler5
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "DARPAHandler5", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "cadets5": DARPAHandler5,
    "atlas": ATLASHandler,
}


def get_handler(name, train, PATH_MAP, scene_name: Optional[str] = None):
    """获取指定数据集处理器。

    参数:
    - name: 数据集名称 (如 cadets / atlas)
    - scene_name: 场景过滤 (如 cadets314)。为 None 时加载该数据集下全部场景。
    """
    lower_name = name.lower()
    cls = handler_map.get(lower_name)
    base_path = PATH_MAP.get(lower_name)

    if cls is None or base_path is None:
        raise ValueError(f"未配置数据路径或未知数据集: {name}")

    return cls(base_path, train, scene_name=scene_name)
