# utils/results_manager.py
from .visualization import plot_feature_importance

class ResultsManager:
    def __init__(self, config):
        self.config = config

    def save_results(self, model_name, model, X):
        feature_importance = model.get_feature_importance(X.columns)
        # Save feature importance to file
        self._plot_feature_importance(model_name, feature_importance)

    def _plot_feature_importance(self, model_name, feature_importance):
        plot_config = self.config.get_nested('visualization', 'feature_importance_plot')
        plot_feature_importance(
            feature_importance,
            title=f"{model_name} Feature Importance - All Tickers Combined",
            figsize=tuple(plot_config['figsize']),
            rotation=plot_config['rotation']
        )
