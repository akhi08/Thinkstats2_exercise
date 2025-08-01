import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__=="__main__":
    df=pd.read_csv('/Users/akhichoudhary/STATS/Thinkstats/Thinkstats2_exercise/Thinkstats/Approach any problem with ml/mnist_classifier/input/mnist_train.csv')
    # Shuffle data
    df=df.sample(frac=1).reset_index(drop=True)
    # Create a new column for fold  assignment
    df['kfold']=-1
    # Initialize stratified k-fold (assuming 'label' column exists)
    skf=StratifiedKFold(n_splits=5)
    # assign folds
    for fold,(train_idx,val_idx) in enumerate(skf.split(X=df,y=df.label)):
        df.loc[val_idx,'kfold']=fold
    #Save the new csv
    df.to_csv("input/mnist_train_folds.csv", index=False)


# StratifiedKFold ensures each fold has roughly the same percentage of each class label.
# df = df.sample(frac=1).reset_index(drop=True)
# - Shuffles the dataset randomly.
# - frac=1 → return 100% of the data, just shuffled.(if 50% then  only 50% will get shuffled)
# - reset_index(drop=True) drops the old index and reassigns a new one
# Before
# Index    label
# 0      0
# 1      0
# 2      1
# 3      1
# 4      2
# 5      2

# After
# Index   label
# 0      2
# 1      0
# 2      1
# 3      2
# 4      0
# 5      1



# df["kfold"] = -1
# - Adds a new column kfold and initializes all values to -1.
# - This will later store the fold number (0 to 4 for 5 folds).


# skf = StratifiedKFold(n_splits=5)
# - Initializes Stratified K-Fold with 5 folds.
# - Ensures each fold has balanced class distribution.


# for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df.label)):
    # df.loc[val_idx, "kfold"] = fold
# - Loops through the 5 folds.
# - skf.split() returns indices for training and validation.
# - Assigns the fold number to the kfold column for validation data only.
# index | label | kfold
# ---------------------
# 0     |   5   |  0
# 1     |   0   |  1
# 2     |   3   |  2
# ...

# 💡 What's happening
# You're looping through each fold created by StratifiedKFold and:
# train_idx → indexes for training rows
# val_idx → indexes for validation rows
# You assign the fold number to the validation rows in the DataFrame


# 🧠 Why use StratifiedKFold?
# It ensures that each fold has a similar distribution of classes (i.e., digits 0–9 in MNIST), 
# avoiding imbalanced validation sets.
# Suppose df.label looks like this after shuffling:
# index | label
# -------------
# 0     |   0
# 1     |   1
# 2     |   2
# 3     |   0
# 4     |   1
# 5     |   2

# With n_splits=3, StratifiedKFold might split as:
# Fold 0: indices [0, 4] (val_idx), others are training
# Fold 1: indices [1, 5] (val_idx), ...
# Fold 2: indices [2, 3] (val_idx), ...


# df.loc[val_idx, "kfold"] = fold
# index | label | kfold
# ----------------------
# 0     |   0   |   0
# 1     |   1   |   1
# 2     |   2   |   2
# 3     |   0   |   2
# 4     |   1   |   0
# 5     |   2   |   1

# So later, when training:
# You use df[df.kfold != i] for training
# And df[df.kfold == i] for validation

# Example
# Initial DataFrame:
#    id  label
# 0   0      0
# 1   1      0
# 2   2      1
# 3   3      1
# 4   4      2
# 5   5      2

# 🔀 Step 2: Shuffle and Add kfold Column
# df = df.sample(frac=1).reset_index(drop=True)
# df["kfold"] = -1

# After shuffling (example output):

#    id  label  kfold
# 0   2      1     -1
# 1   5      2     -1
# 2   1      0     -1
# 3   0      0     -1
# 4   4      2     -1
# 5   3      1     -1

# 🔁 Step 3: Stratified Fold Assignment

# skf = StratifiedKFold(n_splits=3)

# for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df.label)):
#     df.loc[val_idx, "kfold"] = fold


# Now df might look like this (varies due to shuffling but class balance is maintained):

#    id  label  kfold
# 0   2      1      0
# 1   5      2      1
# 2   1      0      2
# 3   0      0      1
# 4   4      2      0
# 5   3      1      2

# | Fold | Validation Samples | Class Distribution in Val |
# | ---- | ------------------ | ------------------------- |
# | 0    | ids 2, 4           | 1, 2                      |
# | 1    | ids 5, 0           | 2, 0                      |
# | 2    | ids 1, 3           | 0, 1                      |

# 🧪 Test: Selecting Folds Later
# To train on fold 0:

# train_df = df[df.kfold != 0]
# valid_df = df[df.kfold == 0]





