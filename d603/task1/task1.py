#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

pd.set_option("display.max_columns", None)  # view all columns when printing
spacer = "*" * 20
# churn = pd.read_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/churn_clean.csv")
# columns_and_dtypes = pd.read_excel("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/task1_dtypes.xlsx")
churn = pd.read_csv("C:/alcibiades_vscode/WGU/d603/task1/churn_clean.csv")
columns_and_dtypes = pd.read_excel(
    "C:/alcibiades_vscode/WGU/d603/task1/task1_dtypes.xlsx"
)


# EDA steps
def EDA(dataframe):
    """
    collection of views for initial look at data
    """
    print(spacer, "HEAD", spacer)
    print(dataframe.head())  # view the data
    print("*" * 20, "DESCRIBE", "*" * 20)
    print(dataframe.describe())  # look at numerical columns
    print("*" * 20, "SHAPE", "*" * 20)
    print(dataframe.shape)  # see number of rows/columns
    print("*" * 20, "ISNULL", "*" * 20)
    print(
        dataframe.isnull().sum()
    )  # look for blank cells (InternetService has 2129 nulls)
    print("*" * 20, "ISDUP", "*" * 20)
    print(dataframe.duplicated().sum())  # look for duplicated rows (no dups)
    print("*" * 20, "DTYPES", "*" * 20)
    print(
        dataframe.dtypes
    )  # look at data types (fireplace, color, garage are the only non-numeric columns, lets look at those)


# fix nulls in InternetService
churn = churn.fillna("No Service")
EDA(churn)

# keeping the variables listed in D2
keep = columns_and_dtypes["Variable"]
churn = churn[keep]
print("*" * 20, "kept variables", "*" * 20)
print(churn)


# encoding churn dataframe
def boolean_encode(dataframe, columns):
    for i in columns:
        print(dataframe[i].head())
        print(f"The data type is {dataframe[i].dtype}")
        dataframe[i] = dataframe[i].replace({"Yes": 1, "No": 0})
    return dataframe


def encoding(dataframe, list_of_columns):
    churn_encoded = pd.get_dummies(dataframe, columns=list_of_columns, dtype=int)
    return churn_encoded


categorical_vars = columns_and_dtypes[columns_and_dtypes["DTYPE"] == "categorical"]
categorical_vars = categorical_vars["Variable"]
print(categorical_vars)

var_bool = []
var_cat = []
for i in categorical_vars:
    unique = set(churn[i])
    if "Yes" in unique:
        var_bool.append(i)
    else:
        var_cat.append(i)
    print(f"The unique values in {i} are {unique}")
# item1-8 do not need encoded
list_items = ["Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8"]
list_toencode = []
for i in var_cat:
    if i not in list_items:
        list_toencode.append(i)
print(f"Boolean variables: {var_bool}")
print(f"Categorical variables: {var_cat}")
print(f"Categorical variables to encode: {list_toencode}")

print("*" * 20, "encoded variables", "*" * 20)
churn_encoded = encoding(churn, list_toencode)
churn_encoded_columns = churn_encoded.columns
print(churn_encoded_columns)  # check for proper one-hot encoding
print(churn_encoded.head())
print("*" * 20, "new code", "*" * 20)
print(churn_encoded_columns)
for i in churn_encoded_columns:
    print(i)
churn_encoded = boolean_encode(churn_encoded, churn_encoded_columns)
print(churn_encoded.head())
print(churn_encoded.dtypes)

# churn_encoded.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/churn_encoded.csv", index=False)
churn_encoded.to_csv(
    "C:/alcibiades_vscode/WGU/d603/task1/churn_encoded.csv", index=False
)

# split the data


def drop_unwanted(dataframe, columns_to_drop):
    dataframe = dataframe.drop(columns_to_drop, axis=1)
    print(dataframe.columns)

    return dataframe


def split_data_70_15_15(X, y):
    """
    Splits data into 85/15 train/test, then splits the 85% into train/validate.

    X: indepentant variables
    y: dependant variable
    """

    # 1. split into train/test
    test_percentage = 0.15
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percentage, random_state=42, stratify=y
    )
    # 2. calculate percentage for second split
    val_percentage = 0.15
    val_percentage = val_percentage / (1 - test_percentage)
    # 2.5. split train into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_percentage, random_state=42, stratify=y_train
    )
    # 3. check sets to make sure the 70/15/15 set is correct
    print(f"Training set has {len(X_train)} records.")
    print(f"Validation set has {len(X_val)} records.")
    print(f"Testing set has {len(X_test)} records.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# export split data for rubric
def export_splits_to_csv(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    exports my data for the grader to review/use
    """
    # 1. rejoin X and y
    df_train = pd.concat([X_train, y_train], axis=1)
    df_validate = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # 2. export dfs
    # df_train.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/df_train.csv", index=False)
    # df_validate.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/df_validate.csv", index=False)
    # df_test.to_csv("C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/df_test.csv", index=False)

    df_train.to_csv("C:/alcibiades_vscode/WGU/d603/task1/df_train.csv", index=False)
    df_validate.to_csv(
        "C:/alcibiades_vscode/WGU/d603/task1/df_validate.csv", index=False
    )
    df_test.to_csv("C:/alcibiades_vscode/WGU/d603/task1/df_test.csv", index=False)


# set dependant (y) and independant (X) variables
y = churn_encoded["Churn"]
X = drop_unwanted(churn_encoded, "Churn")

X_train, X_val, X_test, y_train, y_val, y_test = split_data_70_15_15(X, y)
# export_splits_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)


# train initial model
def display_model_metrics(rcf, X, y, model_name):
    """
    takes model and returns metrics required by rubric
    X/y: either X_val/y_val or X_test/y_test
    """
    # make predictions
    y_pred = rcf.predict(X)
    y_pred_prob = rcf.predict_proba(X)[:, 1]

    # metrics
    print(f"{spacer} Performance Metrics for {model_name} {spacer}")
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")

    print(f"\n{spacer} Confusion Matrix {spacer}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rcf.classes_)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.grid(False)
    # plt.savefig(f"C:/Users/donovan.a.hall.NREN/PycharmProjects/WGU/d603/task1/{model_name}_confusion_matrix")
    plt.savefig(f"C:/alcibiades_vscode/WGU/d603/task1/{model_name}_confusion_matrix")


print(f"{spacer} bug fixing {spacer}")
print(y_train.head(10).apply(lambda x: (x, type(x))))
print(y_train.shape)
y_train = y_train.astype("int64").to_numpy()
y_val = y_val.astype("int64").to_numpy()
y_test = y_test.astype("int64").to_numpy()

initial_model = RandomForestClassifier(random_state=42)
print(f"{spacer} Training Initial Model {spacer}")
initial_model.fit(X_train, y_train)

display_model_metrics(initial_model, X_val, y_val, model_name="Initial_Model")

# hyperparameter tuning
# 1. define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# 2. set variable for best data
best_score = -1.0
best_params = {}
run_counter = 0

# 3. set up hyperparameter loop (uses every possible combination)

print(f"{spacer} Hyperparameter Tuning {spacer}")


# model with this run's parameters
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

best_score = grid_search.best_score_
best_params = grid_search.best_params_

print(f"Best F1 Score: {best_score:.4f}")
print(f"Best Hyperparameters: {best_params}")

tuned_model = grid_search.best_estimator_

# using tuned model on test set
print(f"\n{spacer} Evaluating tuned model on test set {spacer}")
display_model_metrics(tuned_model, X_test, y_test, model_name="Tuned_Model")
