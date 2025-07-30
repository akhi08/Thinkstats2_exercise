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