from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import LeaveOneOut


def csv_save_dir(data_frame, data_path, model_type, dataset):
    cwd = os.getcwd()
    joint_path = os.path.join(cwd, "fs", data_path, model_type+"_"+dataset+".csv")
    data_frame.to_csv(joint_path)


# Script initialization
model = "pls"  # "rfe", "etc", "ust", "pls", "xgb", "lasso", "sbe"
dataset = "cambridge"  # srpbs/opp/yale/cambridge
data = "all"  # all/ str / fc / fun
num_fc = 6216  # number of functional connectivity pairs
num_str = 202  # number of structural features
num_fun = 336  # number of functional features

# Load data from srpbs/opp/yale/cambridge dataset
if dataset == "srpbs":
    df_mri = pd.read_csv("/src/top_features_srpbs_all.csv")
elif dataset == "opp":
    df_mri = pd.read_csv("/src/top_features_opp_all.csv")
elif dataset == "yale":
    df_mri = pd.read_csv("/src/top_features_yale_all.csv")
else:
    df_mri = pd.read_csv("/src/top_features_cambridge_all.csv")

# Data cleansing and extraction of feature names
df_mri.fillna(0)
feature_names = df_mri.columns.values.tolist()
feature_names.pop(0)
mri_arr = df_mri.to_numpy()

# Data selection from all/fc/str/fun
if data == "all":
    mri_data = mri_arr[:, 1:]
elif data == "fc":
    mri_data = mri_arr[:, 1:num_fc]
    feature_names = feature_names[:num_fc-1]
elif data == "str":
    mri_data = mri_arr[:, num_fc+1:num_fc+num_str+1]
    feature_names = feature_names[num_fc:num_fc+num_str]
else:
    mri_data = mri_arr[:, num_fc+num_str+2:]
    feature_names = feature_names[num_fc:num_fc+num_str+1:]

mri_label = mri_arr[:, 0]
scaler = MinMaxScaler()
scaler.fit(mri_data)
mri_data = scaler.transform(mri_data)

# Features selection using Recursive Feature Elimination
if model == "rfe":
    # create the RFE model for the svm classifier
    svm = LinearSVC(dual=0)
    rfe = RFE(svm, 1, 1, verbose=1)
    rfe = rfe.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    print(rfe.support_)
    print(rfe.ranking_)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = rfe.support_
    df.loc['Ranking'] = rfe.ranking_
    csv_save_dir(df, data, model, dataset)

# Features selection using Extra Trees Classifier
elif model == "etc":
    etc = ExtraTreesClassifier(n_estimators=1)
    etc.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    fi = np.array(etc.feature_importances_)
    print(etc.feature_importances_)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = etc.feature_importances_
    df.loc['Ranking'] = len(fi) - rankdata(fi, method='ordinal')
    csv_save_dir(df, data, model, dataset)

# Features selection using Uni-variate Statistical Tests
elif model == "ust":
    ust = SelectKBest(score_func=f_classif, k=50)
    fit = ust.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    np.set_printoptions(precision=3)
    print(fit.scores_)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = fit.scores_
    df.loc['Ranking'] = len(fit.scores_) - rankdata(fit.scores_, method='ordinal')
    csv_save_dir(df, data, model, dataset)

# Features selection using Partial Least Squares regression (PLS)
elif model == "pls":
    # leave one out validation
    cv = LeaveOneOut()
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    mse = []
    for i in range(1, 21):
        pls_opt = PLSRegression(n_components=i)
        score = -1 * model_selection.cross_val_score(pls_opt, mri_data, mri_label, cv=cv,
                                                     scoring='neg_mean_squared_error', verbose=1).mean()
        mse.append(score)

    # plot test MSE vs. number of components
    plt.plot(mse)
    plt.xlabel('Number of PLS Components')
    plt.ylabel('MSE')
    plt.title('Optimal Number of PLS Components')
    plt.show()

    def vip(mod):
        t = mod.x_scores_
        w = mod.x_weights_
        q = mod.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    opt_index = mse.index(min(mse))+1
    print(f"Optimal Number of PLS Component = {opt_index}")
    pls = PLSRegression(n_components=opt_index)
    # pls = PLSRegression(n_components=2)

    rank_n_split = []
    cv.get_n_splits(mri_data)
    i = 0
    for train_index, test_index in cv.split(mri_data):
        X_train, X_test = mri_data[train_index, :], mri_data[test_index, :]
        y_train, y_test = mri_label[train_index], mri_label[test_index]
        pls.fit(X_train, y_train)

        # print summaries for the selection of attributes
        # vip_score = vip(pls)
        # print(vip_score)
        # print(np.abs(pls.coef_[:, 0]))
        feature_ranking = len(np.abs(pls.coef_[:, 0])) - rankdata(np.abs(pls.coef_[:, 0]), method='ordinal')
        if i == 0:
            rank_n_split = feature_ranking
        else:
            rank_n_split = np.vstack([rank_n_split, feature_ranking])
        i += 1
    print(rank_n_split)
    # save as csv
    # df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    # df.loc['Support'] = np.abs(pls.coef_[:, 0])
    # df.loc['Ranking'] = len(np.abs(pls.coef_[:, 0])) - rankdata(np.abs(pls.coef_[:, 0]), method='ordinal')
    # df.loc['Ranking'] = len(vip_score) - rankdata(vip_score, method='ordinal')
    df = pd.DataFrame(rank_n_split)
    df.columns = feature_names
    csv_save_dir(df, data, model, dataset)

# Features selection using XGBoost
elif model == "xgb":
    xgb = XGBRegressor()
    xgb.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    importance = xgb.feature_importances_
    print(xgb.feature_importances_)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = importance
    df.loc['Ranking'] = len(importance) - rankdata(importance, method='ordinal')
    csv_save_dir(data, model, dataset)

# Features selection using least absolute shrinkage and selection operator
elif model == "lasso":
    lasso = linear_model.Lasso(alpha=1,
                               positive=True,
                               fit_intercept=False,
                               max_iter=1000,
                               tol=0.0001)
    lasso.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    print(lasso.coef_)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = np.abs(lasso.coef_)
    df.loc['Ranking'] = len(np.abs(lasso.coef_)) - rankdata(np.abs(lasso.coef_), method='ordinal')
    csv_save_dir(data, model, dataset)

# Features selection using KNeighborsClassifier
elif model == "sbe":
    knn = KNeighborsClassifier(n_neighbors=30)
    sfs = SequentialFeatureSelector(knn, n_features_to_select=30)
    sfs.fit(mri_data, mri_label)

    # print summaries for the selection of attributes
    importance = sfs.get_support()
    print(importance)

    # save as csv
    df = pd.DataFrame(columns=feature_names, index=['Support', 'Ranking'])
    df.loc['Support'] = 1 * importance
    df.loc['Ranking'] = len(importance) - rankdata(importance, method='ordinal')
    csv_save_dir(data, model, dataset)