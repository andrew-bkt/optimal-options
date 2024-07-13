# options_screening/utils/visualization.py

import matplotlib.pyplot as plt

def plot_feature_importance(feature_importance, title="Feature Importance", figsize=(12, 6), rotation=45):
    plt.figure(figsize=figsize)
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()