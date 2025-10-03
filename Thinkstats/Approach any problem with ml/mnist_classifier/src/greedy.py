# greedy.py
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    def evaluate_score(self,X,y):
        model=linear_model.LogisticRegression()
        model.fit(X,y)
        prediction=model.predict_proba(X)[:,1]
        auc=metrics.roc_auc_score(y,predictions)
        return auc

    def _feature_selection(self,X,y):
        good_feature=[]
        best_scores=[]

        num_features=X.shape[1]

        while True:
            this_feature=None
            best_score=0
                