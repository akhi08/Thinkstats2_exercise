#ohe_logres.py  for cat-in data

# import pandas as pd
# from sklearn import linear_model
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
#     # it doesn’t matter because all are categories
#     for col in features:
#         df.loc[:, col] = df[col].astype(str).fillna("NONE")
#     # get training data using folds
#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     # get validation data using folds
#     df_valid = df[df.kfold == fold].reset_index(drop=True)
#     # initialize OneHotEncoder from scikit-learn
#     ohe = preprocessing.OneHotEncoder()
#     # fit ohe on training + validation features
#     full_data = pd.concat(
#     [df_train[features], df_valid[features]],
#     axis=0
#     )
#     ohe.fit(full_data[features])
#     # transform training data
#     x_train = ohe.transform(df_train[features])
#     # transform validation data
#     x_valid = ohe.transform(df_valid[features])
#     # initialize Logistic Regression model
#     model = linear_model.LogisticRegression()
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
# # run function for fold = 0
# # we can just replace this number and
# # run this for any fold
#     for fold_ in range(5):
#         run(fold_)

###################################################################################################################################################

# ohe_logres.py for adult.dataest

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("input/adult_folds.csv")
    # list of numerical columns
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
    features = [c for c in df.columns if c not in ("kfold", "income")]

    # clean income column first
    df["income"] = df["income"].str.strip()
    
    # map targets
    target_mapping = {"<=50K": 0, ">50K": 1}
    df["income"] = df["income"].map(target_mapping)
    
    # get training and validation
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # OneHotEncoder
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")
    
    # fit only on training
    ohe.fit(df_train[features])
    
    # transform
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    
    # model
    model = linear_model.LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(x_train, df_train.income.values)
    
    # predict
    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"Fold={fold}, AUC={auc:.4f}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)