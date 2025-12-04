#tfv_logres.py

import pandas as pd

from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer



if __name__=="__main__":
    df = pd.read_csv("/Users/akhichoudhary/STATS/Thinkstats/Thinkstats2_exercise/Thinkstats/Approach any problem with ml/mnist_classifier/input/All_dataset/imdb.csv")
    df.sentiment=df.sentiment.apply(lambda x: 1 if x=='positive' else 0)

    df['kfold']=-1

    df=df.sample(frac=1).reset_index(drop=True)

    y=df.sentiment.values

    kf=model_selection.StratifiedKFold(n_splits=5)

    for f_,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold']=f_


    for fold_ in range(5):
        train_df=df[df.kfold!=fold_].reset_index(drop=True)

        test_df=df[df.kfold==fold_].reset_index(drop=True)

        tfidf_vec = TfidfVectorizer(tokenizer=word_tokenize,token_pattern=None)
        # fit tfidf_vec on training data reviews
        tfidf_vec.fit(train_df.review)
        # transform training and validation data reviews
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)
        
        
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
# ══════════════════





# for f, (train_indices, validation_indices) in folds:
# you are looping through 5 splits, and unpacking each into two variables.
    # the above code is basically:
        # pairs = [(1, 11), (2, 22), (3, 33)]
        # for x, y in pairs:
        #      print("x:", x, " y:", y)
    # (train_index_array, validation_index_array)
      # [
     # (array([...train...]), array([...valid...])),
     # (array([...train...]), array([...valid...])),
     # (array([...train...]), array([...valid...])),
     # (array([...train...]), array([...valid...])),
     # (array([...train...]), array([...valid...]))
    # ]
    # 
    # pairs = [
    # (train_0, valid_0),
    # (train_1, valid_1),
    # (train_2, valid_2),
    # (train_3, valid_3),
    # (train_4, valid_4),
# ]


# Instead of manually creating folds, use cross_validate and automatically retrieve fold indices.
# from sklearn.model_selection import StratifiedKFold, cross_validate
# from sklearn.linear_model import LogisticRegression

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# model = LogisticRegression()

# cv_results = cross_validate(
#     model,
#     df["review"],
#     df["sentiment"],
#     cv=kf,
#     return_train_score=True,
#     return_estimator=True,
# )


# Using sklearn.model_selection.GroupKFold (Modern industry standard for NLP/Vision)

# If dataset has duplicates / same users / same movies etc.
#     from sklearn.model_selection import GroupKFold

# groups = df["movie_id"]
# gkf = GroupKFold(n_splits=5)

# df["kfold"] = -1

# for f, (_, v_) in enumerate(gkf.split(df, y=df.sentiment, groups=groups)):
#     df.loc[v_, "kfold"] = f

