from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")


def plot_feature_importances(model, X_train, output_path, regressor_id):
    feature_names = X_train.columns

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(f"{output_path}/feature_importances_{regressor_id}.csv")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df.head(10)["feature"], importance_df.head(10)["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top 10 Feature Importances for {regressor_id}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importances_{regressor_id}.png")
    plt.clf()
    plt.close()


def plot_regression_performance(y_test, y_pred, r2, output_path, regressor_id):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    axes[0, 0].set_title(f"{regressor_id}: Actual vs Predicted (R² = {r2:.3f})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title(f"{regressor_id}: Residuals Plot")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(
        f"{regressor_id}: Distribution of Residuals (Mean: {residuals.mean():.4f})"
    )
    axes[1, 0].grid(True, alpha=0.3)

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"{regressor_id}: Q-Q Plot")

    plt.tight_layout()
    plt.savefig(f"{output_path}/regression_performance_{regressor_id}.png")
    plt.clf()
    plt.close()
