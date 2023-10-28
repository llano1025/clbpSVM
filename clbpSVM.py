import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import os
import clbp_lib as cl

# rfe (Recursive Feature Elimination) / etc (Extra Trees Classifier) / ust (Uni-variate Statistical Tests)
# pls (Partial Least Squares)/ xgb (eXtreme Gradient Boosting) / sbe (Sequential Backward Elimination)

# Script initialization
algo_sort = "pls_opp"  # Top features ranked by featureSelection.py
feature = "all"  # all/str/fc/fun
dataset = "all"  # self/opp/yale/cambridge/all
kernel = 'adoptive'  # linear/poly/rbf/sigmoid
num_feature = range(5, 51, 1)  # number of feature for training
num_fc = 6216  # number of functional connectivity pairs
num_str = 202  # number of structural features
num_fun = 336  # number of functional features

# Load data set from csv file
df_srpbs = pd.read_csv("/src/top_features_srpbs_all.csv")
df_opp = pd.read_csv("/src/top_features_opp_all.csv")
df_yale = pd.read_csv("/src/top_features_yale_all.csv")
df_cambridge = pd.read_csv("/src/top_features_cambridge_all.csv")

# Setting list of features
feature_names = df_srpbs.columns.values.tolist()
feature_names.pop(0)

# Load data based on selected options
mri_srpbs, df_fi, feature_names = cl.select_data(feature, df_srpbs, feature_names, num_fc, num_str)
mri_opp = cl.select_data(feature, df_opp, feature_names, num_fc, num_str)[0]
mri_yale = cl.select_data(feature, df_yale, feature_names, num_fc, num_str)[0]
mri_cambridge = cl.select_data(feature, df_cambridge, feature_names, num_fc, num_str)[0]

# Loading columns set from features importance
df_fi = cl.sort_features(df_fi, algo_sort)


# Define data array
class DataList:
    def __init__(self, input_df):
        # Create label array
        self.label = input_df.to_numpy()[:, 0]
        # Create list for result storage
        self.acc_pfold = list()
        self.prc_pfold = list()
        self.rec_pfold = list()
        self.acc_mean_folds = list()
        self.prc_folds = list()
        self.rec_folds = list()
        self.acc_max_folds = list()
        self.acc_min_folds = list()


srpbs = DataList(df_srpbs)
opp = DataList(df_opp)
yale = DataList(df_yale)
cambridge = DataList(df_cambridge)

# Sorting the data frame according to importance
df_srpbs = df_srpbs[df_fi.columns]
df_opp = df_opp[df_fi.columns]
df_yale = df_yale[df_fi.columns]
df_cambridge = df_cambridge[df_fi.columns]
feature_names_sorted = df_fi.columns.values.tolist()

for i in num_feature:

    # defining data array
    def define_data(input_df):
        mri_arr = input_df.to_numpy()
        mri_arr[np.isnan(mri_arr)] = 0
        mri_data = mri_arr[:, 0:i]
        return mri_data

    srpbs_data = define_data(df_srpbs)
    opp_data = define_data(df_opp)
    yale_data = define_data(df_yale)
    cambridge_data = define_data(df_cambridge)
    scaler = MinMaxScaler()
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)

    # clearing list for each fold loop
    def clear_list(x):
        x.acc_pfold.clear()
        x.prc_pfold.clear()
        x.rec_pfold.clear()


    clear_list(srpbs)
    clear_list(opp)
    clear_list(yale)
    clear_list(cambridge)

    if (algo_sort == "pls_srpbs") or (algo_sort == "pls_hku1") or (algo_sort == "pls_hku2") or (algo_sort == "pls_hku3"):
        input_data = srpbs_data
        input_label = srpbs.label
    elif algo_sort == "pls_opp":
        input_data = opp_data
        input_label = opp.label
    elif algo_sort == "pls_yale":
        input_data = yale_data
        input_label = yale.label
    else:  # pls_cambridge
        input_data = cambridge_data
        input_label = cambridge.label

    for train, test in kfold.split(input_data, input_label):

        X_train = input_data[train]
        y_train = input_label[train]

        # Create validation data set
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.11, random_state=1)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)

        svm_linear = {'kernel': ['linear'],
                      'C': [0.1, 1, 10, 100, 1000]}
        svm_poly = {'kernel': ['poly'],
                    'C': [0.1, 1, 10, 100, 1000],
                    'degree': [2, 3, 4, 5, 6, 7, 8],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto']}
        svm_others = {'kernel': ['rbf', 'sigmoid'],
                      'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto']}
        param_grid = [svm_linear, svm_poly, svm_others]
        clf = svm.SVC()
        grid = GridSearchCV(clf, param_grid, refit=True, verbose=3)
        grid.fit(X_valid, y_valid)
        print(grid.best_params_)
        clf = grid.best_estimator_
        # if kernel == 'linear':
        #     clf = svm.SVC(kernel='linear', C=1)
        # elif kernel == 'poly':
        #     clf = svm.SVC(kernel='poly', degree=3, gamma=1, C=0.1)
        # elif kernel == 'rbf':
        #     clf = svm.SVC(kernel='rbf', gamma='auto', C=10)
        clf.fit(X_train, y_train)


        def app_fold_result(y_data, y_label, x):
            X_test = scaler.transform(y_data)
            y_pred = clf.predict(X_test)
            x.acc_pfold.append(metrics.accuracy_score(y_label, y_pred) * 100)
            x.prc_pfold.append(metrics.precision_score(y_label, y_pred) * 100)
            x.rec_pfold.append(metrics.recall_score(y_label, y_pred) * 100)


        if dataset == "self":
            app_fold_result(srpbs_data[test], srpbs.label[test], srpbs)
        elif dataset == "opp":
            app_fold_result(opp_data, opp.label, opp)
        elif dataset == "yale":
            app_fold_result(yale_data, yale.label, yale)
        elif dataset == "cambridge":
            app_fold_result(cambridge_data, cambridge.label, cambridge)
        else:
            app_fold_result(srpbs_data[test], srpbs.label[test], srpbs)
            app_fold_result(opp_data, opp.label, opp)
            app_fold_result(yale_data, yale.label, yale)
            app_fold_result(cambridge_data, cambridge.label, cambridge)

    # provide average scores per number of features
    def app_test_result(x):
        x.acc_mean_folds.append(np.mean(x.acc_pfold))
        x.acc_max_folds.append(np.max(x.acc_pfold) - np.mean(x.acc_pfold))
        x.acc_min_folds.append(np.mean(x.acc_pfold) - np.min(x.acc_pfold))
        x.prc_folds.append(np.mean(x.prc_pfold))
        x.rec_folds.append(np.mean(x.rec_pfold))
        print(x.acc_mean_folds)


    if dataset == "self":
        app_test_result(srpbs)
    elif dataset == "opp":
        app_test_result(opp)
    elif dataset == "yale":
        app_test_result(yale)
    elif dataset == "cambridge":
        app_test_result(cambridge)
    else:
        print('------------------------------------------------------------------------')
        app_test_result(srpbs)
        app_test_result(opp)
        app_test_result(yale)
        app_test_result(cambridge)


# provide average scores
def print_score(x, y, z, m, k):
    print('------------------------------------------------------------------------')
    print('Score per number of features')
    for j in range(0, len(x.acc_mean_folds)):
        print('------------------------------------------------------------------------')
        print(
            f'> No. of features : {j + num_feature[0]} - Accuracy: {x.acc_mean_folds[j]}% - Precision: {x.prc_folds[j]}% - '
            f'Recall: {x.rec_folds[j]}%\n'
            f'> No. of features : {j + num_feature[0]} - Accuracy: {y.acc_mean_folds[j]}% - Precision: {y.prc_folds[j]}% - '
            f'Recall: {y.rec_folds[j]}%\n'
            f'> No. of features : {j + num_feature[0]} - Accuracy: {z.acc_mean_folds[j]}% - Precision: {z.prc_folds[j]}% - '
            f'Recall: {z.rec_folds[j]}%\n'
            f'> No. of features : {j + num_feature[0]} - Accuracy: {m.acc_mean_folds[j]}% - Precision: {m.prc_folds[j]}% - '
            f'Recall: {m.rec_folds[j]}%')
        if j > 0 & (x.acc_mean_folds[j] >= x.acc_mean_folds[j - 1]) & (y.acc_mean_folds[j] >= y.acc_mean_folds[j - 1]):
            k.append(feature_names_sorted[j + num_feature[0]])
    # print(z)
    # df = pd.DataFrame(z, columns=["column"])
    # df.to_csv("F:/SRPBS_DB/working/bp_t/best_features/"+algo_sort+data+"_2.csv", index=False)


best_features = [feature_names_sorted[0]]
print_score(srpbs, opp, yale, cambridge, best_features)


# show the plot
def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = np.max(y)
    text = "x={:.3f}, y={:.3f}%".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def plot_score(x, y, z, k):
    plt.plot(num_feature, x.acc_mean_folds, color='red', marker='.', label="srpbs")
    plt.plot(num_feature, y.acc_mean_folds, color='blue', marker='x', label="opp")
    plt.plot(num_feature, z.acc_mean_folds, color='green', marker='*', label="yale")
    plt.plot(num_feature, k.acc_mean_folds, color='orange', marker='<', label="cambridge")
    # plt.plot(num_feature, x.prc_folds, color='blue', marker='x', label="precision")
    # plt.plot(num_feature, x.rec_folds, color='green', marker='s', label="recall")
    annot_max(num_feature, k.acc_mean_folds)
    plt.title(f"Accuracy Analysis for Number of Features in SVM ({kernel}/{feature}/{algo_sort})")
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()


plot_score(srpbs, opp, yale, cambridge)

# save the result
result_data = {'Accuracy_SRPBS': srpbs.acc_mean_folds,
               'Accuracy_OPP': opp.acc_mean_folds,
               'Accuracy_YALE': yale.acc_mean_folds,
               'Accuracy_CAMBRIDGE': cambridge.acc_mean_folds,
               'Precision_SRPBS': srpbs.prc_folds,
               'Precision_OPP': opp.prc_folds,
               'Precision_YALE': yale.prc_folds,
               'Precision_CAMBRIDGE': cambridge.prc_folds,
               'Recall_SRPBS': srpbs.rec_folds,
               'Recall_OPP': opp.rec_folds,
               'Recall_YALE': yale.rec_folds,
               'Recall_CAMBRIDGE': cambridge.rec_folds,
               'Number of features': num_feature}
result_df = pd.DataFrame(result_data)
result_df.set_index('Number of features', inplace=True)
joint_path = os.path.join("result", feature, algo_sort + "_" + kernel + ".csv")
result_df.to_csv(joint_path)
