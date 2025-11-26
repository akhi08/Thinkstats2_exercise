#ctv_logres.py

import pandas as pd

from ntlk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


if __name__=="__main__":
    # read the training data
    df=pd.read_csv("/Thinkstats/Approach any problem with ml/mnist_classifier/input/All_dataset/imdb.csv")
    df.sentiment=df.sentimenta.apply(
    lambda x: 1 if x=="positive" else 0)

    df['kfold']=-1

    df=df.sample(frac=1).reset_index(drop=True)

    y=sentiment.values

    kf=model_selection.Straf