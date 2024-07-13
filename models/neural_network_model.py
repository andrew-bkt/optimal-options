from .base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

class NeuralNetworkModel(BaseModel):
    def __init__(self, X, y, params):
        super().__init__(X, y)
        self.params = params
        self.scaler = StandardScaler()

    def build_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        if len(self.X) == 0:
            raise ValueError("No data available for training. Check your data processing steps.")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = self.build_model(X_train.shape[1])
        
        history = self.model.fit(X_train_scaled, y_train, 
                                 validation_split=0.2,
                                 epochs=50, 
                                 batch_size=32, 
                                 verbose=0)
        
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return (self.model.predict(X_scaled) > 0.5).astype(int)

    def get_feature_importance(self, feature_names):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        first_layer_weights = np.abs(self.model.layers[0].get_weights()[0])
        feature_importance = np.sum(first_layer_weights, axis=1)
        return pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values('importance', ascending=False)
