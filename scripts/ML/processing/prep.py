from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

import pandas as pd

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def perform_imputation(X_train, X_test):
    discrete_cols = []
    continuous_cols = []

    for col in X_train.columns:
        n_unique = X_train[col].nunique(dropna=True)
        if (n_unique <= 10) or (X_train[col].dtype in ['int64', 'int32', 'int16', 'int8', 'bool']):
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)

    print(f"    {len(discrete_cols)} discrete and {len(continuous_cols)} continuous features")

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    if discrete_cols:
        print(f"    Imputing discrete features ({len(discrete_cols)} columns)...")
        disc_imputer = SimpleImputer(strategy='most_frequent')
        
        X_train_discrete = disc_imputer.fit_transform(X_train[discrete_cols])
        X_test_discrete = disc_imputer.transform(X_test[discrete_cols])
        
        X_train_imputed[discrete_cols] = X_train_discrete
        X_test_imputed[discrete_cols] = X_test_discrete

    if continuous_cols:
        print(f"    Imputing continuous features ({len(continuous_cols)} columns) with IterativeImputer...")
        cont_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42,
            initial_strategy='mean'
        )
        X_train_continuous = cont_imputer.fit_transform(X_train[continuous_cols])
        X_test_continuous = cont_imputer.transform(X_test[continuous_cols])
        
        X_train_imputed[continuous_cols] = X_train_continuous
        X_test_imputed[continuous_cols] = X_test_continuous

    train_nan = X_train_imputed.isna().sum().sum()
    test_nan = X_test_imputed.isna().sum().sum()

    if train_nan == 0 and test_nan == 0:
        print(" Imputation successful! No missing values remain.")
    else:
        print(f"    Warning: {train_nan} train and {test_nan} test NaN remain")

    return X_train_imputed, X_test_imputed


def split_and_prepare(heliad_data, target_var, feature_list):
    """
    Preparation of and split into train- and test sets for ML application.
    A few preparatory steps are included, such as data imputation based on whether
    features are continuous or discrete after the split has been performed (to avoid data leakage)
    and scaling.
    """
    print(f"1. Dropping NA's from the target var {target_var}") 
    heliad_data.dropna(subset=[target_var], inplace=True)

    print(f"2. Splitting the data to X and Y")
    X = heliad_data[feature_list]
    Y = heliad_data[target_var]

    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
    
    print(f"    Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"    Number of features: {X_train.shape[1]}")

    print("3. Imputation of missing values on splitted train and test sets")
    X_train_imputed, X_test_imputed = perform_imputation(X_train, X_test)

    print("4. Scale Features")
    X_train_scaled, X_test_scaled = scale_data(X_train=X_train_imputed, X_test=X_test_imputed)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("All Done!")
    
    return X_train, X_test, y_train, y_test

    
