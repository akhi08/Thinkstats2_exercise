import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

# Method - 1
# Where some code is hard coded,like path to csv ,where to store the model etc....

# def run(fold):
#     # read the training data with folds
#     df = pd.read_csv("/Users/akhichoudhary/STATS/Thinkstats/Thinkstats2_exercise/Thinkstats/Approach any problem with ml/mnist_classifier/input/mnist_train_folds.csv")
#     # training data is where kfold is not equal to provided fold
#     # also, note that we reset the index
#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     # validation data is where kfold is equal to provided fold
#     df_valid = df[df.kfold == fold].reset_index(drop=True)
#     # drop the label column from dataframe and convert it to
#     # a numpy array by using .values.
#     # target is label column in the dataframe
#     x_train = df_train.drop("label", axis=1).values
#     y_train = df_train.label.values
#     # similarly, for validation, we have
#     x_valid = df_valid.drop("label", axis=1).values
#     y_valid = df_valid.label.values
#     # initialize simple decision tree classifier from sklearn
#     clf = tree.DecisionTreeClassifier()
#     # fit the model on training data
#     clf.fit(x_train, y_train)
#     # create predictions for validation samples
#     preds = clf.predict(x_valid)

#     # calculate & print accuracy
#     accuracy=metrics.accuracy_score(y_valid,preds)
#     print(f"Fold={fold},Accuracy={accuracy}")
#     # save the model
#     joblib.dump(clf, f"/Users/akhichoudhary/STATS/Thinkstats/Thinkstats2_exercise/Thinkstats/Approach any problem with ml/mnist_classifier/models/dt_{fold}.bin")


# if __name__ == "__main__":
#     run(fold=0)
#     run(fold=1)
#     run(fold=2)
#     run(fold=3)
#     run(fold=4)

# Method- 2
# removing the hard code and making it more re-usable
import os
import config
# we have created this file to store different path so as to avoid hard code.
import joblib

def run(fold):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values
    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model
    joblib.dump(
                clf,os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
                )
# if __name__ == "__main__":
#     run(fold=0)
#     run(fold=1)
#     run(fold=2)
#     run(fold=3)
#     run(fold=4)


# training script that can be improved. As
# you can see, we call the run function multiple times for every fold. Sometimes it’s
# not advisable to run multiple folds in the same script as the memory consumption
# may keep increasing, and your program may crash. To take care of this problem,
# we can pass arguments to the training script. I like doing it using argparse.

# if we don't use the above "argparse" the command in .sh file will loop.

import argparse

if __name__=="__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument("--fold",type=int)
    # read the arguments from the command line
    args = parser.parse_args()
    # run the fold specified by command line arguments
    run(fold=args.fold)

# Now, we can run the python script again, but only for a given fold.
# ❯ python src/train.py --fold 0

# or using
# And you can run this by the following command.
# ═════════════════════════════════════════════════════════════════════════
# ❯ sh run_all.sh

