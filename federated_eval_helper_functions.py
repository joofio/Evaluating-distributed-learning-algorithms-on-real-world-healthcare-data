import re
import sys
import warnings
from collections import Counter, defaultdict

import federated_eval_helper_functions
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.stats.api as sms
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, RandomOverSampler
from IPython.display import Audio
from matplotlib import pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from pylab import cm
from scipy.stats import ttest_ind
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    plot_confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
import statistics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from itertools import compress
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression, SGDClassifier


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)



    
    
def create_target(y, op="encoder"):
    if op == "binarizer":
        l = LabelBinarizer()
        #    print(y_pipe.fit_transform(y))
        yy = l.fit_transform(y)
    elif op == "encoder":
        # label
        l = LabelEncoder()
        yy = l.fit_transform(y.astype(str))
    else:
        raise ("unkown op")
    return yy, l





def create_analysis_silos(silos, cat_cols, int_cols, drop_cols):

    result = {}
    for idx, silo in enumerate(silos):
        silo = silo.copy().drop(columns=drop_cols)
        col_list = silo.columns
        result["silo" + str(idx + 1)] = []
        for col in silo.columns:
            #  print(col)
            type_col = "categorical" if col in cat_cols else "continuous"
            result["silo" + str(idx + 1)].append(get_summary(silo, col, type_col))
    return result, col_list

class myvotingClassifier:  # estimators,X,voting="soft",weigths=None,limit=0.5
    def __init__(self, estimators, voting="soft", weights=None, threshold=0.5):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.threshold = threshold

    def set_weigths(self, w):
        self.weights = w

    def __repr__(self):
        return "myvotingClassifier(voting='" + self.voting + "')"

    def __str__(self):
        return "instance of myvotingClassifier"

    def get_weigths(self):
        return self.w

    def f(self, x):
        return 1 if x >= self.threshold else 0

    def array_for(self, x):
        return np.array([self.f(xi) for xi in x])

    def predict(self, X):
        preds = np.zeros((len(self.estimators), len(X)))
        for idx, est in enumerate(self.estimators):
            if self.voting == "soft":
                preds[idx, :] = est.predict_proba(X)[:, 1]
            else:  # hard
                preds[idx, :] = est.predict(X)[:, 1]

        avg = np.average(preds, axis=0, weights=self.weights)

        result = self.array_for(avg)
        return result

    def predict_proba(self, X):
        preds = np.zeros((len(self.estimators), len(X)))
        avg = np.average(preds, axis=0, weights=self.weights)
        return avg

    def tune_estimator_weights(
        self, X_train_list, y_train_list, scoring="accuracy", n_repeats=10
    ):
        result = []
        i = 0
        for idx in range(len(X_train_list)):
            for clf in self.estimators:
                yy, l = create_target(y_train_list[idx], op="binarizer")
                scores = accuracy_score(
                    clf.predict(X_train_list[idx]), y_train_list[idx]
                )
                result.append(scores)
                i += scores
        self.w = [v / i for v in result]

    def set_threshold(self, threshold):
        self.threshold = threshold
        
def create_global_model(models, method, w=None):
    if method == "average":
        r_intercepts = []
        r_coefs = []
        r_weights = []
        unique_classes = models[0].classes_
        for l_model in models:
            r_intercepts.append(l_model.intercept_)
            r_coefs.append(l_model.coef_)
            r_weights.append(1 / len(models))
        g_coef_ = np.zeros((1, len(silo1prep.columns)))

        for _ in range(len(models)):
            g_intercept_ = np.average(r_intercepts, axis=0, weights=r_weights)
            g_coef_ = np.average(r_coefs, axis=0, weights=r_weights)
            global_model = set_weights(g_intercept_, g_coef_, unique_classes)
    elif method == "voting":
        global_model = myvotingClassifier(models, weights=w)
        lr = LogisticRegression()
        eclf = EnsembleVoteClassifier(clfs=models, weights=w, fit_base_estimators=False)
        sclf = StackingClassifier(
            classifiers=models,
            meta_classifier=lr,
            fit_base_estimators=False,
            use_probas=True,
            average_probas=False,
        )
    else:  # partial fit
        global_model = None
    return {"myvoting": global_model, "ensemble": eclf, "stacking": sclf}


def get_best_threshold(model, X_list, y_list):
    best = 0
    b_threshold = 0.5
    for threshold in np.arange(0, 1, 0.01):
        avg_list = []
        for X, y in zip(X_list, y_list):
            try:
                avg_list.append(roc_auc_score(model.predict(X, threshold), y))
            except:
                avg_list.append(np.nan)
        avg = np.mean(avg_list)
        if avg > best:
            best = avg
            b_threshold = threshold
    return b_threshold, best


def define_weights(grid_list):
    i = 0
    result_list = []
    for grid in grid_list:
        result_list.append(grid.best_score_)
        i += grid.best_score_
    return [v / i for v in result_list]


def get_stats(total, print_val=False,):
    final = {}
    total_values = []
    for k, v in total.items():
        for p, t in v.items():
            if p not in ["models", "g_model"]:
                mean = np.nanmean(t)
                median = np.median(t)
                #   ci = sms.DescrStatsW(t).tconfint_mean()
                ci = st.t.interval(
                    alpha=0.95,
                    df=len(t) - 1,
                    loc=np.nanmean(t),
                    scale=st.sem(t, nan_policy="omit"),
                )
                if print_val:
                    print(k, p, mean, median, ci)
                data = np.array(t)
                if np.isnan(data).sum() / len(data) < 0.3:

                    filtered_data = data[~np.isnan(data)]
                    total_values.append(filtered_data)
                else:
                    total_values.append(t)

                final[k + "_" + p] = [mean, ci, median, t]
    return final

def get_ttest(x):
    final = {}
    for k, d in x.items():
        final[k] = []
        mylist = [t for k, t in d.items() if k not in ["models", "g_model"]]
        labels = [k for k, t in d.items() if k not in ["models", "g_model"]]
        l = labels[1::2]
        zipped = list(zip(mylist[0::2], mylist[1::2]))
        i = 0
        for t in zipped:

            ttest, pval = ttest_ind(t[0], t[1], nan_policy="omit")
            #     print(pval)
            final[k].append(
                {
                    "name": l[i][0:-7],
                    "mean1": np.nanmean(t[0]),
                    "mean2": np.nanmean(t[1]),
                    "pvalue": pval,
                    "outcome": "equal" if pval >= 0.05 else "not equal",
                }
            )
            i += 1
    return final



#deal with missing classes and low frequency classes:
#low frequency - smote
#non-existing: create dummy with median/most frequent 
def dummy_row_creation(
    data,target,target_value,
    int_cols,
    cat_cols,nr_rows
):
    o_col_list=data.columns
    for i in range(nr_rows):
        data=data.append(pd.Series(), ignore_index=True)
    data.loc[data[target].isna(),target]=target_value
    
    numeric_transformer = Pipeline(
            steps=[
                ("imputer1", SimpleImputer(strategy="median", missing_values=np.nan))
            ]
        )

    categorical_transformer = Pipeline(
            steps=[ 
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=np.nan, strategy="most_frequent"
                    ))]
        )

    preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, int_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    
    XX = pipeline.fit_transform(data)
    col_list = (
        pipeline["preprocessor"].transformers_[0][2] #changes col location
        + pipeline["preprocessor"].transformers_[1][2]
    )
    df1 = pd.DataFrame(XX, columns=col_list)
    #reorder columns
    df1=df1[o_col_list]
    return df1


def prepare_global_model(g_model, X_train, y_train):
    if type(g_model) == StackingClassifier:
        return g_model.fit(X_train, y_train)
    elif type(g_model) == EnsembleVoteClassifier:
        return g_model.fit(X_train, y_train)
    else:
        threshold = get_best_threshold(global_model, [X_train], [y_train])
        g_model.tune_estimator_weights([X_train], [y_train], scoring="roc_auc_score")
        g_model.set_threshold(threshold[0])
        return g_model