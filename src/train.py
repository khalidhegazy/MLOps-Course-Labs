import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib


def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    churn_maj, churn_min = (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
    churn_maj_downsample = resample(churn_maj, n_samples=len(churn_min), replace=False, random_state=1234)
    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    filter_feat = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited"
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]

    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough"
    )

    X_train_transf = col_transf.fit_transform(X_train)
    X_test_transf = col_transf.transform(X_test)

    # Save the preprocessor for inference
    joblib.dump(col_transf, "preprocessor.pkl")

    return X_train_transf, X_test_transf, y_train, y_test


def train_and_save_models(X_train, X_test, y_train, y_test):
    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    print("\nLogistic Regression:")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("F1 Score:", f1_score(y_test, y_pred_log))

    # Save Logistic Regression model
    joblib.dump(log_reg, "logistic_model.pkl")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\nRandom Forest:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("F1 Score:", f1_score(y_test, y_pred_rf))

    # Save the better model
    joblib.dump(rf, "model.pkl")  # You can use this in FastAPI


def main():
    df = pd.read_csv("../dataset/Churn_Modelling.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    train_and_save_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
