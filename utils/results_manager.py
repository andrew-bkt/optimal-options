# utils/results_manager.py
from .visualizations import (plot_feature_importance, plot_confusion_matrix, plot_roc_curve,
                             plot_precision_recall_curve, plot_learning_curve, plot_model_comparison)
from sklearn.model_selection import learning_curve
import numpy as np

class ResultsManager:
    def __init__(self, config):
        self.config = config
        self.results = {}

    def save_results(self, model_name, model, X, y):
        cv_results = model.cross_validate(cv=self.config.get_nested('cross_validation', 'n_folds', 5))
        self.results[model_name] = {
            'model': model,
            'feature_importance': model.get_feature_importance(X.columns),
            'predictions': model.predict(X),
            'probabilities': model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None,
            'cv_results': cv_results
        }
        self._plot_results(model_name, X, y)

    def _plot_results(self, model_name, X, y):
        result = self.results[model_name]
        plot_feature_importance(result['feature_importance'], title=f"{model_name} - Feature Importance")
        plot_confusion_matrix(y, result['predictions'], classes=['0', '1'], title=f"{model_name} - Confusion Matrix")
        if result['probabilities'] is not None:
            plot_roc_curve(y, result['probabilities'], title=f"{model_name} - ROC Curve")
            plot_precision_recall_curve(y, result['probabilities'], title=f"{model_name} - Precision-Recall Curve")
        
        # Generate learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            result['model'].model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
        plot_learning_curve(train_sizes, train_scores, test_scores, title=f"{model_name} - Learning Curve")

        result = self.results[model_name]
        cv_results = result['cv_results']
        print(f"\n--- {model_name} Cross-Validation Results ---")
        print(f"Mean CV Score: {cv_results['cv_mean_score']:.4f} (+/- {cv_results['cv_std_score']:.4f})")
        print("\nClassification Report:")
        print(cv_results['cv_report'])
        print("\nConfusion Matrix:")
        print(cv_results['cv_confusion_matrix'])

    def plot_model_comparison(self, metric='cv_mean_score'):
        model_names = list(self.results.keys())
        scores = [result['cv_results'][metric] for result in self.results.values()]
        plot_model_comparison(model_names, scores, f'Cross-Validated {metric.replace("_", " ").title()}')

    def print_summary(self):
        for model_name, result in self.results.items():
            print(f"\n--- {model_name} Summary ---")
            print("Top 10 Important Features:")
            print(result['feature_importance'].head(10))
            print("\nCross-Validation Results:")
            cv_results = result['cv_results']
            print(f"Mean CV Score: {cv_results['cv_mean_score']:.4f} (+/- {cv_results['cv_std_score']:.4f})")




    def plot_model_comparison(self, metric='accuracy'):
        model_names = list(self.results.keys())
        scores = [result['model'].score(metric) for result in self.results.values()]
        plot_model_comparison(model_names, scores, metric)

    def print_summary(self):
        for model_name, result in self.results.items():
            print(f"\n--- {model_name} Summary ---")
            print("Top 10 Important Features:")
            print(result['feature_importance'].head(10))
            print("\nModel Performance:")
            print(result['model'].performance_report())
