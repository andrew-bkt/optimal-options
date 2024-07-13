# models/model_factory.py
from .random_forest_model import RandomForestModel
from .gradient_boosting_model import GradientBoostingModel
from .neural_network_model import NeuralNetworkModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, X, y, params):
        if model_type == "RandomForestClassifier":
            return RandomForestModel(X, y, params)
        elif model_type == "GradientBoostingClassifier":
            return GradientBoostingModel(X, y, params)
        elif model_type == "NeuralNetworkModel":
            return NeuralNetworkModel(X, y, params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
