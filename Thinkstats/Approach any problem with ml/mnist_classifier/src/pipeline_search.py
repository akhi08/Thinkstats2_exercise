# Sometimes, you might want to use a pipeline. For example, let’s say that we are
# dealing with a multiclass classification problem. In this problem, the training data
# consists of two text columns, and you are required to build a model to predict the
# class. Let’s assume that the pipeline you choose is to first apply tf-idf in a semi-
# supervised manner and then use SVD with SVM classifier. Now, the problem is we
# have to select the components of SVD and also need to tune the parameters of SVM

#pipeline_search.py

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn