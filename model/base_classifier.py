from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
    @abstractmethod
    def train(self) -> None:
        pass
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass
    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict:
        pass
