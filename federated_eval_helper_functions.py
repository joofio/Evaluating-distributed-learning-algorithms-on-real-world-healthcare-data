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
import os
import zipfile

import glob
import pickle
import json
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

from datetime import datetime

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# helper functions federeated eval


def plot_paper_grade_error(final):

    label_l = [
        x.replace("_local", "").replace("_global", "").replace("roc_auc_score", "auc")
        for x in list(final.keys())
    ]
    # label_l=label[0::2]
    metrics = ["_".join(l.split("_")[-1:]) for l in label_l]
    col_name = ["_".join(l.split("_")[0:-2]) for l in label_l]
    metric_silo = ["_".join(l.split("_")[-2:]) for l in label_l]
    nr_plots = len(set(col_name))
    if nr_plots == 1:
        return plot_better_error(
            final
        )  # workaround pq n m apetece rever isto dos multiplos plots
    figheight = 5 * len(set(metrics)) * nr_plots
    fig, axs = plt.subplots(nr_plots, 1, figsize=(12, figheight))
    x_ticks_1 = metric_silo[0::2]
    x_ticks_2 = metric_silo[1::2]
    mean = [x[0] for x in final.values()]
    x_1 = mean[0::2]
    x_2 = mean[1::2]
    cis = [x[1][1] - x[0] for x in final.values()]
    length = len(set(metric_silo))
    #    print(length)
    err_1 = cis[0::2]
    err_2 = cis[1::2]
    l_col_name = np.unique(np.array(col_name))

    for idx, ax in enumerate(axs):
        ax.set_title("Scores by silo for " + str(l_col_name[idx]) + " 95% CI", pad=-3)

        ax.spines["top"].set_visible(False)
        ax.set_ylabel("Silo and metric", labelpad=1)
        ax.set_xlabel("Metric value", labelpad=-1)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_ticks_position("both")

        l_err_1 = err_1[length * idx : length * (idx + 1)]
        l_err_2 = err_2[length * idx : length * (idx + 1)]

        l_x_1 = x_1[length * idx : length * (idx + 1)]
        l_x_2 = x_2[length * idx : length * (idx + 1)]

        ax_height = len(l_x_1)
        l_y_1 = np.linspace(0, ax_height, len(l_x_1))
        l_y_2 = l_y_1 + 0.25
        l_y_label = l_y_1 + 0.12

        l_x_ticks_1 = x_ticks_1[length * idx : length * (idx + 1)]
        l_x_ticks_2 = x_ticks_2[length * idx : length * (idx + 1)]

        ax.set_xlim(-0.05, 1.03)
        ax.set_ylim(-1, ax_height + 3)
        b1 = ax.errorbar(
            x=l_x_1,
            y=l_y_1,
            xerr=l_err_1,
            color="orange",
            capsize=3,
            linestyle="None",
            marker="s",
            markersize=4,
            mfc="orange",
            mec="orange",
        )

        g1 = ax.errorbar(
            x=l_x_2,
            y=l_y_2,
            xerr=l_err_2,
            color="blue",
            capsize=3,
            linestyle="None",
            marker="s",
            markersize=4,
            mfc="blue",
            mec="blue",
        )
        #    print(l_x_ticks_1)
        ax.set_yticks(l_y_label)
        ax.set_yticklabels(l_x_ticks_1)
        ax.tick_params(labelright=False)
        ax.yaxis.grid(True, which="major", linestyle="--", color="gray", alpha=0.5)
        # ax.yaxis.grid(True, which='minor')
        ax.legend([b1, (b1, g1)], ["Local", "Global"], loc=2)
        ax.axvline(0.50, ls="--")
        plt.savefig("teste.png", dpi=300, transparent=False, bbox_inches="tight")
    # plt.show()
    plt.show()


def log_to_file(f, content):
    dt = datetime.now()
    formatted_str = dt.strftime("%Y-%m-%d %H:%M:%S")

    c = [str(con) for con in content]
    f.write(formatted_str + " - ")

    f.write(" ".join(c) + "\n")


def create_mega_table(total, method):
    total_rec = []
    for k, v in total.items():
        rec = {"method": method, "column": k}
        for k2, v2 in v.items():

            if "model" not in k2:
                #  print(k,k2,v2)
                mean = np.nanmean(v2)
                rec[k2 + "_mean"] = mean
                rec[k2 + "_sd"] = np.nanstd(v2)

        total_rec.append(rec)
    df = pd.DataFrame.from_dict(total_rec)
    value_vars = list(
        compress(df.columns.to_list(), ["silo" in k for k in df.columns.to_list()])
    )
    df = pd.melt(df, id_vars=["method", "column"], value_vars=value_vars)

    def get_metric(x):
        if "f1" in x:
            return "F1"
        if "auc" in x:
            return "AUC"
        if "auprc" in x:
            return "AUPRC"

    df["metric"] = df["variable"].apply(get_metric)

    def get_silo(x):
        return x[4]

    df["silo"] = df["variable"].apply(get_silo)

    def get_stat(x):
        return x.split("_")[-1]

    df["stat"] = df["variable"].apply(get_stat)

    def get_model(x):
        return x.split("_")[-2]

    df["model"] = df["variable"].apply(get_model)

    df.drop(columns=["variable"], inplace=True)
    df2 = df.pivot(
        index=["method", "column", "metric", "stat", "model"],
        columns="silo",
        values="value",
    ).reset_index()
    df3 = df2.pivot(
        index=["method", "column", "metric", "model"],
        columns="stat",
        values=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    ).reset_index()
    newcol_name = df3.columns.to_list()

    def create_col_name(x):
        if x[1] in ["mean", "sd"]:
            return "silo_" + x[0] + "_" + x[1]
        else:
            return x[0]

    newercol_name = [create_col_name(v) for v in newcol_name]

    df3.columns = newercol_name
    return df3


def from_dict_to_df_raw(total, method):
    total_rec = []
    for k, v in total.items():
        target = k
        for k2, v2 in v.items():
            ilo_metric_model = k2
            l = k2.split("_")
            silo = l[0]
            metric = "_".join(l[1:-1])
            model = l[-1]
            # print(silo,metric,model)
            for idx, experiment in enumerate(v2):
                row = [target, silo, metric, model, method, idx + 1, experiment]
                # print(experiment)
                # print(row)
                total_rec.append(row)
    xx = pd.DataFrame.from_dict(total_rec)
    xx.columns = ["target", "silo", "metric", "model", "method", "experiment", "value"]
    return xx


def save_zipped_model(target, model_name, model_type, model):
    filename = str(target) + "_" + str(type(model_name).__name__) + "_" + model_type
    # save model
    pickle.dump(model, open(filename + ".pickle", "wb"))
    # Create a zip file
    with zipfile.ZipFile("models/" + filename + ".zip", "w") as zip:
        # Add the pickle file to the zip file
        zip.write(filename + ".pickle")

    os.remove(filename + ".pickle")


def load_model_from_zip(filename):
    # open the zip file in read mode
    with zipfile.ZipFile("models/" + filename + ".zip", "r") as zip_ref:
        # extract the files to the current working directory
        # zip_ref.extract()
        zip_ref.extract(filename + ".pickle")

    # open the pickle file and load the data
    with open(filename + ".pickle", "rb") as f:
        model = pickle.load(f)
    os.remove(filename + ".pickle")
    return model


def create_multirow_latex(cell):
    n_c = cell.replace("_", " ").lower()
    if len(cell) > 20:
        return (
            "\\begin{tabular}{@{}l@{}}"
            + n_c[0:10]
            + "\\\\"
            + n_c[10:]
            + "\\end{tabular}"
        )
    else:
        return n_c


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


def create_pipeline_with_y(
    data,
    target,
    int_cols,
    cat_cols,
    drop_cols=None,
    pipeline=None,
    op="encoder",
    as_df=False,
):
    _cat_cols = [colc for colc in cat_cols if colc not in [target]]
    _int_cols = [coli for coli in int_cols if coli not in [target]]
    n_df = data.copy().drop(columns=drop_cols + [target])

    def to_object(x):
        return pd.DataFrame(x).astype(str)

    def to_number(x):
        return pd.DataFrame(x).astype(float)

    if pipeline == None:
        fun_str = FunctionTransformer(to_object)
        fun_num = FunctionTransformer(to_number)

        numeric_transformer = Pipeline(
            steps=[
                ("fun_num", fun_num),
                ("imputer1", SimpleImputer(strategy="median", missing_values=np.nan)),
                ("imputer2", SimpleImputer(strategy="median", missing_values=-1)),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("fun_str", fun_str),
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=np.nan, strategy="constant", fill_value="NULLIMP"
                    ),
                ),
                ("ordinalEncoder", OrdinalEncoder()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, _int_cols),
                ("cat", categorical_transformer, _cat_cols),
            ]
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        y = data[target]
        X = n_df
        y_pipe = make_pipeline(
            SimpleImputer(
                missing_values=np.nan, strategy="constant", fill_value="NULLIMP"
            )
        )
        yy, l = create_target(y, op=op)
        XX = pipeline.fit_transform(X)

        if as_df == True:
            col_list = (
                pipeline["preprocessor"].transformers_[0][2]  # changes col location
                + pipeline["preprocessor"].transformers_[1][2]
            )
            print(len(col_list))
            df1 = pd.DataFrame(XX, columns=col_list)
            r = pd.concat([df1, pd.DataFrame(yy)], axis=1)
            r.rename(columns={0: target}, inplace=True)
            return r, None, pipeline, l
        return XX, yy, pipeline, l


def preprocessing_df(
    df,
    categorical_columns,
    integ_colums,
    drop_cols=["G_TERAPEUTICA", "IDENTIFICADOR"],
    pipeline=["imputer", "encoder"],
):
    if len(df.columns) != len(categorical_columns) + len(integ_colums) + len(drop_cols):
        # raise Exception
        print("WARNING! Columns number different")
    df1 = df.copy()

    def simple_imputer_categorical_df(df, columns):
        imputer = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value="NULLIMP"
        )  # está a por NULL em variaveis com nrs(apesar de considerar cat)
        imputer.fit(df[columns])
        X = pd.DataFrame(imputer.transform(df[columns]))
        return X

    def simple_imputer_numeric_df(df, columns):
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        imputer.fit(df[columns])
        X = pd.DataFrame(imputer.transform(df[columns]))
        return X

    def scikit_one_hot_encoder(df, categorical_columns):
        enc_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc_hot.fit(df[categorical_columns].astype(str))
        X = pd.DataFrame(
            enc_hot.transform(df[categorical_columns].astype(str)),
            columns=enc_hot.get_feature_names(df[categorical_columns].columns),
        )
        data_encoded = pd.concat([df, X], axis=1)
        data_encoded.drop(columns=df[categorical_columns].columns, inplace=True)
        return data_encoded

    df1.drop(columns=drop_cols, inplace=True)
    if "imputer" in pipeline:
        df1[categorical_columns] = simple_imputer_categorical_df(
            df1, categorical_columns
        )
        df1[integ_colums] = simple_imputer_numeric_df(df1, integ_colums)

    if "encoder" in pipeline:
        df1 = scikit_one_hot_encoder(df1, categorical_columns)

    return df1


def get_summary(silo, col, type_col):
    if type_col == "categorical":
        if silo[col].isna().sum() == len(silo):
            return str("Null [100%]")
        mode = silo[col].mode()
        freq = silo[col].value_counts()[mode].values[0]
        perc = freq / len(silo[col])
        if len(str(mode.values[0])) > 6:
            # return str(mode.values[0])[0:5].lower() +" .. "+str(mode.values[0])[-3:].lower()+ " [" + str(round(perc * 100, 1)) + "%]"
            return (
                str(mode.values[0])[0:5].lower()
                + ".. ["
                + str(round(perc * 100, 0))
                + "%]"
            )

        else:
            return str(mode.values[0]).lower() + " [" + str(round(perc * 100, 0)) + "%]"
    # return " [" + str(round(perc * 100, 1)) + "%]"

    if type_col == "continuous":
        mean = silo[col].mean()
        std = silo[col].std()
        return str(round(mean, 1)) + " (" + str(round(std, 1)) + ")"


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
    # print([m.best_params_ for m in models])
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
        # lr = LogisticRegression()
        eclf = EnsembleVoteClassifier(clfs=models, weights=w, fit_base_estimators=False)
    # sclf = StackingClassifier(
    #     classifiers=models,
    #     meta_classifier=lr,
    #     fit_base_estimators=False,
    #     use_probas=True,
    #     average_probas=False,
    # )
    else:  # partial fit
        global_model = None
    # return {"myvoting": global_model, "ensemble": eclf, "stacking": sclf}

    return {"myvoting": global_model, "ensemble": eclf}


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


def get_stats(
    total,
    print_val=False,
):
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


def plot_error(final):
    plt.figure(figsize=(8, 20))
    for k, v in final.items():
        plt.plot((v[1][0], v[1][1]), (k, k), "ro-", color="orange")


def plot_boxplot(final):
    plt.figure(figsize=(8, 20))

    label = [k for k in final.keys()]
    median = [v[0] for v in final.values()]
    ci = [v[1] for v in final.values()]
    plt.boxplot(
        total_values,
        notch=False,
        conf_intervals=np.array(ci),
        usermedians=median,
        vert=False,
        labels=label,
    )


def plot_better_error(final):
    plt.figure(figsize=(8, 20))

    label_l = [
        x.replace("_local", "").replace("_global", "").replace("roc_auc_score", "auc")
        for x in list(final.keys())
    ]
    #  metrics=["_".join(l.split("_")[-1:]) for l in label_l]
    # silo_nr=["_".join(l.split("_")[-2:-1]) for l in label_l]
    col_name = ["_".join(l.split("_")[0:-2]) for l in label_l]
    metric_silo = ["_".join(l.split("_")[-2:]) for l in label_l]
    # print(col_name)
    height = len(label_l) // 12
    plt.figure(figsize=(12, 8 + height))
    #   plt.title('Scores by target, model and silo')
    ax = plt.axes()
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.5, 18)
    # ax.spines['right'].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Target with Model", labelpad=3)
    ax.set_xlabel("Metric value", labelpad=3)
    # ax.set_xticklabels(['0','2.5','5','7.5','1'])
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

    # plt.show()
    x_ticks_1 = label_l[0::2]
    x_ticks_2 = label_l[1::2]

    mean = [x[0] for x in final.values()]
    x_1 = mean[0::2]
    x_2 = mean[1::2]
    cis = [x[1][1] - x[0] for x in final.values()]
    y_1 = np.arange(0, len(x_1))
    print(y_1)
    y_2 = y_1 + 0.18
    print(y_2)

    err_1 = cis[0::2]
    err_2 = cis[1::2]

    b1 = plt.errorbar(
        x=x_1,
        y=y_1,
        xerr=err_1,
        color="black",
        capsize=3,
        linestyle="None",
        marker="s",
        markersize=4,
        mfc="black",
        mec="black",
    )

    g1 = plt.errorbar(
        x=x_2,
        y=y_2,
        xerr=err_2,
        color="gray",
        capsize=3,
        linestyle="None",
        marker="s",
        markersize=4,
        mfc="gray",
        mec="gray",
    )
    plt.yticks(y_1, x_ticks_1)
    plt.yticks(y_2, x_ticks_2)
    plt.legend([b1, (b1, g1)], ["Local", "Global"], loc=2)
    plt.tight_layout()
    ax.yaxis.grid(True, which="major", linestyle="--", color="gray", alpha=0.5)

    plt.axvline(0.50, ls="--")
    # plt.show()
    plt.savefig("teste.png", dpi=300, transparent=False, bbox_inches="tight")


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


# deal with missing classes and low frequency classes:
# low frequency - smote
# non-existing: create dummy with median/most frequent
def dummy_row_creation(data, target, target_value, int_cols, cat_cols, nr_rows):
    o_col_list = data.columns
    for i in range(nr_rows):
        data = data.append(pd.Series(), ignore_index=True)
    data.loc[data[target].isna(), target] = target_value

    numeric_transformer = Pipeline(
        steps=[("imputer1", SimpleImputer(strategy="median", missing_values=np.nan))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
        ]
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
        pipeline["preprocessor"].transformers_[0][2]  # changes col location
        + pipeline["preprocessor"].transformers_[1][2]
    )
    df1 = pd.DataFrame(XX, columns=col_list)
    # reorder columns
    df1 = df1[o_col_list]
    return df1


def evaluate_variables_and_transform_variables(
    df, target, int_cols, cat_cols, class_list=None, threshold=12, nr_rows=25
):
    if class_list:
        missing_class = list(set(class_list) - set(df[target].unique()))
        if len(missing_class) > 0:
            print("missing class:", missing_class)
            for c in missing_class:
                df = dummy_row_creation(df, target, c, int_cols, cat_cols, nr_rows)
    s = df[target].value_counts().le(threshold)
    to_smote = list(s[s].index.values)
    #   print("vars to be enhanced",to_smote)
    # transform the dataset
    y = df[target]
    X = df.drop(columns=[target])
    if len(to_smote) > 0:
        smote_params = {}
        for e in to_smote:
            smote_params[e] = nr_rows
        ros = RandomOverSampler(random_state=0, sampling_strategy=smote_params)

        X_resampled, y_resampled = ros.fit_resample(X, y)
        # if not present
        return X_resampled, y_resampled
    else:
        return X, y


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
