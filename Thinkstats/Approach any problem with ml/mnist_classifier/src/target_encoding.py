# target_encoding.py
# One more way of feature engineering from categorical features is to use target encoding.
# you must be very careful
# when using target encoding as it is too prone to overfitting. When we use target
# encoding, it’s better to use some kind of smoothing or adding noise in the encoded
# values. Scikit-learn has contrib repository which has target encoding with
# smoothing, or you can create your own smoothing. Smoothing introduces some
# kind of regularization that helps with not overfitting the model.

# ========    used mean. You can use mean, median, standard deviation or any other function of targets. ==========*******
# what's going on 💡
# - The dataset of 100 rows is split into 5 folds → each fold has 80 train + 20 valid rows.
# - For each fold, we compute the mean of Income grouped by categorical column on the train set.
# - This encoding is then mapped onto the 20 validation rows and stored in an encoded dataset.
# - After all 5 folds, the encoded dataset has 100 rows (all validation parts combined), ready for the run function. ✅


# target_encoding.py
import copy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def mean_target_encoding(data):
    # make a copy of dataframe
    df = copy.deepcopy(data)
    # list of numerical columns
    num_cols = [
    "fnlwgt",
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
    "<=50K": 0,
    ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    # all columns are features except income and kfold columns
    features = [f for f in df.columns if f not in ("kfold", "income") and f not in num_cols]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
    # do not encode the numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label encode the features
    for col in features:
        if col not in num_cols:
    # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
    # fit label encoder on all data
            lbl.fit(df[col])
    # transform all the data
            df.loc[:, col] = lbl.transform(df[col])
    # a list to store 5 validation dataframes
    encoded_dfs = []
    # go over all folds
    for fold in range(5):
    # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
    # for all feature columns, i.e. categorical columns
        for column in features:
        # create dict of category:mean target
            mapping_dict = dict(df_train.groupby(column)["income"].mean())
            # column_enc is the new column we have with mean encoding
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
            # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
        # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df
    
def run(df, fold):
    # note that folds are same as before
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # all columns are features except income and kfold columns
    features = [f for f in df.columns if f not in ("kfold", "income")]
    # scale training data
    x_train = df_train[features].values
    # scale validation data
    x_valid = df_valid[features].values
    # initialize xgboost model
    model = xgb.XGBClassifier(n_jobs=-1,max_depth=7)
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.astype(int).values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.astype(int).values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    # read data
    df = pd.read_csv("input/adult_folds.csv")
    # create mean target encoded categories and
    # munge data
    df = mean_target_encoding(df)
    # run training and validation for 5 folds
    for fold_ in range(5):
        run(df, fold_)


# 🧠
# -If you had taken the mean of the validation set and then used it to encode the training set, it would be like “peeking into the exam answers before writing the exam” → the model would see information from data it should not have access to during training.
# - That leakage means:
# - The train set would be influenced by future/hidden info.
# - Model would show inflated performance on validation but fail on unseen/test data.
# - This is classic data leakage → overfitting.
# - 👉 Correct way: Always calculate encoding from train only, then apply to valid/test
