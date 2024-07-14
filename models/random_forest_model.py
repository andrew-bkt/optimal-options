from .base_model import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class RandomForestModel(BaseModel):
    def __init__(self, X, y, params):
        super().__init__(X, y)
        self.params = params

    def train(self):
        if len(self.X) == 0:
            raise ValueError("No data available for training. Check your data processing steps.")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return pd.DataFrame({'feature': feature_names, 'importance': self.model.feature_importances_}).sort_values('importance', ascending=False)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict_proba(X)
