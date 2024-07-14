# models/base_model.py
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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

    def cross_validate(self, cv=5):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv)
        
        # Get cross-validated predictions
        cv_predictions = cross_val_predict(self.model, self.X, self.y, cv=cv)
        
        # Calculate performance metrics
        cv_report = classification_report(self.y, cv_predictions)
        cv_confusion_matrix = confusion_matrix(self.y, cv_predictions)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean_score': np.mean(cv_scores),
            'cv_std_score': np.std(cv_scores),
            'cv_report': cv_report,
            'cv_confusion_matrix': cv_confusion_matrix
        }
