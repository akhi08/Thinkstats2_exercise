# we look at our training script, we
# still are limited by a few things, for example, the model. The model is hardcoded in
# the training script, and the only way to change it is to modify the script. So, we will
# create a new python script called model_dispatcher.py. 

from sklearn import tree
from sklearn import ensemble
models={
    "decision_tree_gini":tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy":tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "random_forest":ensemble.RandomForestClassifier(),
}