from abc import ABC, abstractmethod
import casadi as ca
import numpy as np

class Geometry(ABC):
    @property
    @abstractmethod
    def location(self):
        raise NotImplementedError
    
    def 