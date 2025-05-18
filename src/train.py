"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from mlflow.models.signature import infer_signature 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    import mlflow.sklearn


    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Infer model signature (input/output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(log_reg, "logistic_regression_model", signature=signature)

    # Log the training data shape
    mlflow.log_metric("train_samples", X_train.shape[0])
    mlflow.log_metric("train_features", X_train.shape[1])

    return log_reg


def main():
    ### Set the tracking URI and experiment name
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run():

        df = pd.read_csv("../dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log preprocessing step
        mlflow.sklearn.log_model(col_transf, "preprocessor")

        # Train model and log it
        model = train(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute and log metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log tag
        mlflow.set_tag("developer", "kHALID") 

        # Confusion Matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        conf_mat_disp.plot()
        
        # Save and log confusion matrix plot
        plt.title("Confusion Matrix")
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        plt.show()


if __name__ == "__main__":
    main()