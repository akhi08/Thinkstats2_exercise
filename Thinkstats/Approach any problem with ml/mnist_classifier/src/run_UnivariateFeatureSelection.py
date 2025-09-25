import pandas as pd
from UnivariateFeatureSelection import UnivariateFeatureSelection

# Example dataset
X = pd.DataFrame({
    "a": [1, 2, 3, 4, 5],
    "b": [2, 3, 4, 5, 6],
    "c": [5, 4, 3, 2, 1]
})
y = [1, 3, 2, 5, 4]

# Use your custom selector
ufs = UnivariateFeatureSelection(n_features=0.5, problem_type="regression", scoring="f_regression")
ufs.fit(X, y)
X_transformed = ufs.transform(X)

print("Transformed Data:")
print(X_transformed)



# ============== running another way ===========

# import pandas as pd
# from sklearn.datasets import load_diabetes

# # Example dataset
# data = load_diabetes()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = data.target

# # Select top 30% features for regression using f_regression
# ufs = UnivariateFeatureSelection(n_features=0.3, problem_type="regression", scoring="f_regression")
# X_transformed = ufs.fit_transform(X, y)

# print("Original shape:", X.shape)
# print("Transformed shape:", X_transformed.shape)
