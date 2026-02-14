from .write_performance_stats import write_performance_stats
from .plot_results import plot_feature_importances, plot_regression_performance
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error


def ols_regressor(X_train, y_train, X_test, y_test, output_path):
    """ """
