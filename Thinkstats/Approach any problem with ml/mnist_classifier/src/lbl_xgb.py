#lbl_xgb.py   for cat-in-dat
# =====================================================================================================================================


# import pandas as pd
# import xgboost as xgb
# from sklearn import metrics
# from sklearn import preprocessing

# def run(fold):
#     # load the full training data with folds
#     df = pd.read_csv("input/cat-in-data/cat_train_folds.csv")
#     # all columns are features except id, target and kfold columns
#     features = [
#     f for f in df.columns if f not in ("id", "target", "kfold")
#     ]
#     # fill all NaN values with NONE
#     # note that I am converting all columns to "strings"
#     # it doesnt matter because all are categories
#     for col in features:
#         df.loc[:, col] = df[col].astype(str).fillna("NONE")
#         # now it’s time to label encode the features
#     for col in features:
#         # initialize LabelEncoder for each feature column
#         lbl = preprocessing.LabelEncoder()
#         # fit label encoder on all data
#         lbl.fit(df[col])
#         # transform all the data
#         df.loc[:, col] = lbl.transform(df[col])
#         # get training data using folds
#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     # get validation data using folds
#     df_valid = df[df.kfold == fold].reset_index(drop=True)
#     # get training data
#     x_train = df_train[features].values
#     # get validation data
#     x_valid = df_valid[features].values
#     # initialize xgboost model
#     model = xgb.XGBClassifier(
#         n_jobs=-1,
#         max_depth=7,
#         n_estimators=200
#         )
#     # fit model on training data (ohe)
#     model.fit(x_train, df_train.target.values)
#     # predict on validation data
#     # we need the probability values as we are calculating AUC
#     # we will use the probability of 1s
#     valid_preds = model.predict_proba(x_valid)[:, 1]
#     # get roc auc score
#     auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
#     # print auc
#     print(f"Fold = {fold}, AUC = {auc}")
# if __name__ == "__main__":
#     for fold_ in range(5):
#         run(fold_)


# =====================================================================================================================================
# lbl_xgb.py for adult data set

# lbl_xgb.py
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing
def run(fold):
    # load the full training data with folds
    df = pd.read_csv("input/adult_folds.csv")
    # list of numerical columns
    num_cols = [
    "fnlwgt",
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
    ]
    # drop numerical columns
    df = df.drop(num_cols, axis=1)
    # map targets to 0s and 1s
    target_mapping = {
    "<=50K": 0,
    ">50K": 1
    }
    df.loc[:, "income"] = df.income.str.strip().map(target_mapping)
    # all columns are features except kfold & income columns
    features = [
    f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label encode the features
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()
        # fit label encoder on all data
        lbl.fit(df[col])
        # transform all the data
        df.loc[:, col] = lbl.transform(df[col])
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize xgboost model
    model = xgb.XGBClassifier(
    n_jobs=-1
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.astype(int).values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)