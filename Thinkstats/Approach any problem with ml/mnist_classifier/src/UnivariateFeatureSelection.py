from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile



class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection models from
        scikit-learn.
        :param n_features: SelectPercentile if float else SelectKBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        # for a given problem type, there are only
        # a few valid scoring methods
        # you can extend this with your own custom
        # methods if you wish
        if problem_type == "classification":
            valid_scoring = {
                            "f_classif": f_classif,
                            "chi2": chi2,
                            "mutual_info_classif": mutual_info_classif
                            }
        else:
            valid_scoring = {
                            "f_regression": f_regression,
                            "mutual_info_regression": mutual_info_regression
                            }
        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        # please note that it is int in both cases in sklearn
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                             valid_scoring[scoring],
                             k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                            valid_scoring[scoring],
                            percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")
            # same fit function
    def fit(self, X, y):
        return self.selection.fit(X, y)
    # same transform function
    def transform(self, X):
        return self.selection.transform(X)
    # same fit_transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)




# ==============================Another way==============================

# from sklearn.feature_selection import (
#     chi2, f_classif, f_regression,
#     mutual_info_classif, mutual_info_regression,
#     SelectKBest, SelectPercentile
# )


# class UnivariateFeatureSelection:
#     """
#     A simple wrapper around scikit-learn's univariate feature selection.

#     - Use int for `n_features` → SelectKBest
#     - Use float (0.0 - 1.0) → SelectPercentile
#     - Supports both classification and regression problems
#     """

#     def __init__(self, n_features, problem_type="classification", scoring="f_classif"):
#         self.n_features = n_features
#         self.problem_type = problem_type
#         self.scoring = scoring
#         self.selection = self._init_selector()

#     def _init_selector(self):
#         """Choose the correct scoring function and selector."""
#         scoring_methods = {
#             "classification": {
#                 "f_classif": f_classif,
#                 "chi2": chi2,
#                 "mutual_info_classif": mutual_info_classif,
#             },
#             "regression": {
#                 "f_regression": f_regression,
#                 "mutual_info_regression": mutual_info_regression,
#             },
#         }

#         if self.problem_type not in scoring_methods:
#             raise ValueError("problem_type must be 'classification' or 'regression'")

#         if self.scoring not in scoring_methods[self.problem_type]:
#             raise ValueError(f"Invalid scoring method: {self.scoring}")

#         score_func = scoring_methods[self.problem_type][self.scoring]

#         # Select method based on type of n_features
#         if isinstance(self.n_features, int):
#             return SelectKBest(score_func, k=self.n_features)
#         elif isinstance(self.n_features, float):
#             return SelectPercentile(score_func, percentile=int(self.n_features * 100))
#         else:
#             raise TypeError("n_features must be int (k best) or float (percentile)")

#     def fit(self, X, y):
#         return self.selection.fit(X, y)

#     def transform(self, X):
#         return self.selection.transform(X)

#     def fit_transform(self, X, y):
#         return self.selection.fit_transform(X, y)
