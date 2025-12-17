import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def handle_outliers(X):
    X = pd.DataFrame(X)
    feature_has_outliers = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

    for col in feature_has_outliers:
        if col not in X.columns:
            continue

        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X[col] = X[col].clip(lower, upper)

    return X


def preprocessing_data(
    data_path,
    target_column,
    pipeline_path,
    output_train_path,
    output_test_path,
    header_path,
):

    ensure_dir(pipeline_path)
    ensure_dir(output_train_path)
    ensure_dir(output_test_path)
    ensure_dir(header_path)

    data = pd.read_csv(data_path)

    numeric_features = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = data.select_dtypes(include=["object"]).columns.tolist()

    numeric_features.remove(target_column)

    # Simpan header fitur
    pd.DataFrame(columns=data.columns).to_csv(header_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {header_path}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("outlier", FunctionTransformer(handle_outliers)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    feature_names = numeric_features + list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df[target_column] = y_train.values

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df[target_column] = y_test.values

    # Simpan hasil preprocessing
    pd.DataFrame(train_df).to_csv(output_train_path, index=False)
    print(f"Data train berhasil disimpan di: {output_train_path}")

    pd.DataFrame(test_df).to_csv(output_test_path, index=False)
    print(f"Data test berhasil disimpan di: {output_test_path}")

    dump(preprocessor, pipeline_path)
    print(f"Pipeline berhasil disimpan di: {output_test_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    PROCESSED_DATA_DIR = "preprocessing/heart_failure_preprocessing"
    os.makedirs(os.path.join(BASE_DIR, PROCESSED_DATA_DIR))

    preprocessing_data(
        data_path=os.path.join(BASE_DIR, "heart_failure_raw.csv"),
        target_column="HeartDisease",
        pipeline_path=os.path.join(
            BASE_DIR,
            "preprocessing",
            "heart_failure_preprocessing",
            "preprocessor.joblib",
        ),
        output_train_path=os.path.join(
            BASE_DIR, "preprocessing", "heart_failure_preprocessing", "X_train.csv"
        ),
        output_test_path=os.path.join(
            BASE_DIR, "preprocessing", "heart_failure_preprocessing", "X_test.csv"
        ),
        header_path=os.path.join(
            BASE_DIR,
            "preprocessing",
            "heart_failure_preprocessing",
            "feature_header.csv",
        ),
    )
