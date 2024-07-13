# models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_feature_importance(self, feature_names):
        pass
