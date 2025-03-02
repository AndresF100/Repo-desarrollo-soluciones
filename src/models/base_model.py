
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass
