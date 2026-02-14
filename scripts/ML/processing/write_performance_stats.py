from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def write_performance_stats(
    y_train, y_pred_train, y_test, y_pred, regressor_id, output_path
):

    print(f"Calculating metrics for {regressor_id}...")

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred)

    with open(f"{output_path}/performance_stats_{regressor_id}.txt", "w") as of:
        of.write("Training Performance\n")
        of.write("_" * 60)
        of.write("\n")
        of.write(f"RMSE:  {train_rmse:.4f}\n")
        of.write(f"MSE:   {train_mse:.4f}\n")
        of.write(f"MAE:   {train_mae:.4f}\n")
        of.write(f"R²:    {train_r2:.4f}\n")

        of.write("-" * 60)
        of.write("\n")

        of.write("Prediction Performance\n")
        of.write("_" * 60)
        of.write("\n")
        of.write(f"RMSE:  {test_rmse:.4f}\n")
        of.write(f"MSE:   {test_mse:.4f}\n")
        of.write(f"MAE:   {test_mae:.4f}\n")
        of.write(f"R²:    {test_r2:.4f}\n")
    return test_r2, train_r2
