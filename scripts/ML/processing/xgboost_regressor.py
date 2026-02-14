from .write_performance_stats import write_performance_stats
from .plot_results import plot_feature_importances, plot_regression_performance
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error

import xgboost as xgb


def xgboost_regressor(X_train, y_train, X_test, y_test, output_path):
    format_gap = "\t"
    print("1. XGBoost Regressor")
    print("1.1 Hyperparameter tuning...")

    param_grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.5, 0.6, 0.7],
        "colsample_bytree": [0.5, 0.6, 0.7],
        "reg_alpha": [0.1, 1, 10],
        "reg_lambda": [1, 10, 50],
    }
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    xgb_tuned = xgb.XGBRegressor(
        n_estimators=500,
        random_state=42,
        max_delta_step=1,
        n_jobs=-1,
        early_stopping_rounds=30,
        verbosity=0,
    )
    grid_search = GridSearchCV(
        xgb_tuned,
        param_grid,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_fit, y_train_fit, eval_set=[(X_val, y_val)])
    print(f"{format_gap}\nBest parameters:", grid_search.best_params_)
    print(f"{format_gap}Best cross-validation score:", -grid_search.best_score_)

    print("2. Perform predictions with the best model")

    best_model = grid_search.best_estimator_

    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    if hasattr(best_model, "best_iteration") and best_model.best_iteration is not None:
        actual_rounds = best_model.best_iteration
        print(f"Early stopping stopped at {actual_rounds} trees")

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"3. Calculate performance metrics and write into file under {output_path}")
    _, test_r2 = write_performance_stats(
        y_train, y_pred_train, y_test, y_pred_test, "XGBoost", output_path
    )

    print(
        f"4. Extract feature importances and save plot and scores under {output_path}"
    )
    plot_feature_importances(best_model, X_train, output_path, "XGBoost")

    print(f"5. Plot regression performance and save plot under {output_path}")
    plot_regression_performance(y_test, y_pred_test, test_r2, output_path, "XGBoost")

    print("\n6. Overfitting Analysis:")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Gap ratio:  {test_rmse/train_rmse:.2f}x (lower is better)")

    print("=" * 40)
