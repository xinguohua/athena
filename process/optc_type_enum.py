# object_type_enum.py
from enum import Enum

class optcObjectType(Enum):
    PROCESS = 0
    THREAD = 1
    FILE = 2
    FLOW = 3
    MODULE = 4
    TASK = 5
    REGISTRY = 6
    USER_SESSION = 7
    SHELL = 8
