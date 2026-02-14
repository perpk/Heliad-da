from typing import Final
from enum import Enum
from processing import split_and_prepare, initialize_and_execute_regressor
import json
import pandas as pd
import os
from pathlib import Path

OUTPUT_PATH: Final = "./ml_output"
HELIAD_FILE: Final = "data_pre_proc.csv"
TARGET_VAR: Final = "ZCO"


class FeatureSelectionMethod(Enum):
    LASSO = "LASSO_selected_features.json"
    KBest = "SelectKBest_selected_features.json"

    def get_path(self):
        script_dir = Path(__file__).parent.absolute()
        return script_dir / self.value


def main():
    regressors = ["OLS"]  # "XGBOOST"]
    selected_features = []
    featureSelectionMethod = FeatureSelectionMethod.LASSO

    print("=" * 80)
    print(f"Using features selected by {featureSelectionMethod}")

    with open(featureSelectionMethod.get_path()) as feat:
        selected_features = json.load(feat)

    hf = Path(__file__).parent.absolute() / HELIAD_FILE
    print(f"Reading Heliad study file from {hf}")
    df = pd.read_csv(hf, header=0)
    print(f"Dimensions of the Heliad study data file: {df.shape}")

    X_train, X_test, y_train, y_test = split_and_prepare(
        df, TARGET_VAR, selected_features
    )
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for regressor in regressors:
        initialize_and_execute_regressor(
            regressor, X_train, X_test, y_train, y_test, OUTPUT_PATH
        )


if __name__ == "__main__":
    main()
