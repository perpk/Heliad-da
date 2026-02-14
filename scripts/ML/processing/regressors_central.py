from .xgboost_regressor import xgboost_regressor
from time import time


def initialize_and_execute_regressor(
    regressor_id, X_train, X_test, y_train, y_test, output_path
):
    start = int(time() * 1000)
    if regressor_id == "XGBOOST":
        xgboost_regressor(X_train, y_train, X_test, y_test, output_path)

    finished_in_sec = int(time() * 1000) - start
    print(
        f"{regressor_id} regression analysis terminated after {finished_in_sec/1000} seconds."
    )
