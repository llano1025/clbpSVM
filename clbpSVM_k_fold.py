import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import clbp_lib as cl


# rfe (Recursive Feature Elimination) / etc (Extra Trees Classifier) / ust (Uni-variate Statistical Tests)
# pls (Partial Least Squares)/ xgb (eXtreme Gradient Boosting) / sbe (Sequential Backward Elimination)

# Script initialization
find_best_parameter = False
algo_sort = "pls"  # Top features ranked by featureSelection.py
dataset = "opp"  # self/opp
feature = "all"  # all/ str / fc / fun
num_fc = 6216  # number of functional connectivity pairs
num_str = 202  # number of structural features
num_fun = 336  # number of functional features
num_feature = 30
num_folds = range(2, 21, 1)
kernel = "rbf"

# load data set from csv file
df_srpbs = pd.read_csv("/src/top_features_srpbs_all.csv")
df_opp = pd.read_csv("/src/top_features_opp_all.csv")
df_yale = pd.read_csv("/src/top_features_yale_all.csv")

# setting list of features
feature_names = df_srpbs.columns.values.tolist()
feature_names.pop(0)

# data selection
mri_srpbs, df_fi, feature_names = cl.select_data(feature, df_srpbs, feature_names, num_fc, num_str)
mri_opp = cl.select_data(feature, df_opp, feature_names, num_fc, num_str)[0]
mri_yale = cl.select_data(feature, df_yale, feature_names, num_fc, num_str)[0]

# Loading columns set from features importance
df_fi = cl.sort_features(df_fi, algo_sort)

# defining label array
mri_arr = df_srpbs.to_numpy()
opp_arr = df_opp.to_numpy()
mri_label = mri_arr[:, 0]
opp_label = opp_arr[:, 0]

df_srpbs = df_srpbs[df_fi.columns]
df_opp = df_opp[df_fi.columns]

# defining data array
feature_names_sorted = df_fi.columns.values.tolist()
mri_arr = df_srpbs.to_numpy()
opp_arr = df_opp.to_numpy()
# df_opp.fillna(0)
mri_arr[np.isnan(mri_arr)] = 0
opp_arr[np.isnan(opp_arr)] = 0
mri_data = mri_arr[:, 0:num_feature]
opp_data = opp_arr[:, 0:num_feature]

# # plot permutation test score
# clf = SVC(kernel="linear", random_state=7)
# cv = StratifiedKFold(2, shuffle=True, random_state=0)
# score_mri, perm_scores_mri, pvalue_mri = permutation_test_score(
#     clf, X_train, y_train, scoring="accuracy", cv=cv, n_permutations=5000
# )
# print("Score:", score_mri)
# print("Perm Score:", perm_scores_mri)
# print("pValue:", pvalue_mri)
#
# fig, ax = plt.subplots()
# ax.hist(perm_scores_mri, bins=20, density=True)
# ax.axvline(score_mri, ls="--", color="r")
# score_label = f"Score on original\ndata: {score_mri:.2f}\n(p-value: {pvalue_mri:.3f})"
# ax.text(0.7, 10, score_label, fontsize=12)
# ax.set_xlabel("Accuracy score")
# ax.set_ylabel("Probability")
# plt.show()

# leave one out validation
# loo = LeaveOneOut()
# loo.get_n_splits(mri_data)
# y_true, y_pred = list(), list()
# for train_index, test_index in loo.split(mri_data):
#     X_train, X_test = mri_data[train_index, :], mri_data[test_index, :]
#     y_train, y_test = mri_label[train_index], mri_label[test_index]
#     clf = svm.SVC(kernel='linear')
#     clf.fit(X_train, y_train)
#     yhat = clf.predict(X_test)
#     y_true.append(y_test[0])
#     y_pred.append(yhat[0])
#
# acc = accuracy_score(y_true, y_pred)
# print('Accuracy: %.3f' % acc)

# k-fold validation
acc_pfold, prc_pfold, rec_pfold = list(), list(), list()
acc_mean_folds, prc_folds, rec_folds = list(), list(), list()
acc_max_folds, acc_min_folds = list(), list()
coef_folds = np.zeros((num_feature,))

for k in num_folds:

    kfold = KFold(n_splits=k, random_state=1, shuffle=True)
    acc_pfold.clear()
    prc_pfold.clear()
    rec_pfold.clear()

    for train, test in kfold.split(mri_data, mri_label):

        # normalize the data and create a svm classifier
        scaler = MinMaxScaler()
        scaler.fit(mri_data[train])
        X_train = scaler.transform(mri_data[train])
        if kernel == 'linear':
            clf = svm.SVC(kernel='linear', C=1)
        elif kernel == 'poly':
            clf = svm.SVC(kernel='poly', degree=2, gamma=1, C=0.1)
        elif kernel == 'rbf':
            clf = svm.SVC(kernel='rbf', gamma='auto', C=10)
        clf.fit(X_train, mri_label[train])
        if k == 10 and kernel == "linear":
            # coef_folds = np.vstack((coef_folds, np.abs(clf.coef_[0])))
            coef_folds = np.vstack((coef_folds, clf.coef_[0]))

        # Generate generalization metrics
        if dataset == "self":
            X_test = scaler.transform(mri_data[test])
            y_pred = clf.predict(X_test)
            acc_pfold.append(metrics.accuracy_score(mri_label[test], y_pred) * 100)
            prc_pfold.append(metrics.precision_score(mri_label[test], y_pred) * 100)
            rec_pfold.append(metrics.recall_score(mri_label[test], y_pred) * 100)
        else:
            X_test = scaler.transform(opp_data)
            y_pred = clf.predict(opp_data)
            acc_pfold.append(metrics.accuracy_score(opp_label, y_pred) * 100)
            prc_pfold.append(metrics.precision_score(opp_label, y_pred) * 100)
            rec_pfold.append(metrics.recall_score(opp_label, y_pred) * 100)

    # provide average scores per fold
    acc_mean_folds.append(np.mean(acc_pfold))
    acc_max_folds.append(np.max(acc_pfold) - np.mean(acc_pfold))
    acc_min_folds.append(np.mean(acc_pfold) - np.min(acc_pfold))
    prc_folds.append(np.mean(prc_pfold))
    rec_folds.append(np.mean(rec_pfold))

    print('------------------------------------------------------------------------')
    print(f'Score per fold w/ total fold number = {k}')
    for i in range(0, len(acc_pfold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Accuracy: {acc_pfold[i]}% - Precision: {prc_pfold[i]}% - Recall: {rec_pfold[i]}%')
    print('------------------------------------------------------------------------')
    print(f'Training for k-fold number = {k} ...')
    print(f'> Accuracy: {np.mean(acc_pfold)}% (+- {np.std(acc_pfold)}%)')
    print(f'> Precision: {np.mean(prc_pfold)}% (+- {np.std(prc_pfold)}%)')
    print(f'> Recall: {np.mean(rec_pfold)}% (+- {np.std(rec_pfold)}%)')

# provide average scores
print('------------------------------------------------------------------------')
print('Score per folds')
for i in range(0, len(acc_mean_folds)):
    print('------------------------------------------------------------------------')
    print(
        f'> No. of fold : {i + num_folds[0]} - Accuracy: {acc_mean_folds[i]}% - Precision: {prc_folds[i]}% - Recall: {rec_folds[i]}%')

# show the plot
plt.errorbar(num_folds, acc_mean_folds, yerr=[acc_min_folds, acc_max_folds], fmt='o', ecolor='r', color='b')
plt.title(f"Sensitivity Analysis for k number in SVM ({feature}/{algo_sort})")
plt.xlabel('k-number')
plt.ylabel('Accuracy(%)')
plt.show()


# determine most contributing features
def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = list(zip(*sorted(zip(imp, names), reverse=True)))
    print(names)

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][-top:], align='center')
    plt.yticks(range(top), names[::-1][-top:])
    for index, value in enumerate(imp[::-1][-top:]):
        plt.text(value, index,
                 str(value))
    plt.title(f"SVM Feature Importance ({algo_sort}/10 k-folds)")
    plt.show()

    def csv_save_dir(data_path, model_type):
        joint_path = os.path.join("fs", data_path + model_type + "_top10.csv")
        df.to_csv(joint_path)

    df = pd.DataFrame(columns=names, index=['Weights'])
    df.loc['Weights'] = imp
    csv_save_dir(feature, algo_sort)


if kernel == 'linear':
    rank = coef_folds.mean(axis=0)
    f_importances(rank, feature_names_sorted[:num_feature], top=num_feature)

# Finding the best estimator

if find_best_parameter:

    best_parameter = list()

    for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(mri_data, mri_label, test_size=0.3, random_state=i)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        opp_data = scaler.transform(opp_data)

        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'],
                      'degree': [2, 3, 4, 5, 6, 7, 8],
                      'kernel': ['linear', 'poly', 'rbf']}

        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        best_parameter.append(grid.best_params_)
        print('------------------------------------------------------------------------')
        grid_predictions = grid.predict(X_test)
        print(classification_report(y_test, grid_predictions))
        print('------------------------------------------------------------------------')
        grid_predictions_opp = grid.predict(opp_data)
        print(classification_report(opp_label, grid_predictions_opp))

    dict = {'parameter': best_parameter}
    df = pd.DataFrame(dict)
    df.to_csv('/result/best_parameter.csv')
