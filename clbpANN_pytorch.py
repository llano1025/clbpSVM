import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import clbp_lib as cl


class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class network(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, hidden1_size)
        # self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_out = nn.Linear(hidden1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(hidden1_size)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        # x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

# rfe (Recursive Feature Elimination) / etc (Extra Trees Classifier) / ust (Uni-variate Statistical Tests)
# pls (Partial Least Squares)/ xgb (eXtreme Gradient Boosting) / sbe (Sequential Backward Elimination)


# Script initialization
algo_sort = "pls_opp"  # Top features ranked by featureSelection.py
feature = "all"  # all/ str / fc / fun
dataset = "all"  # self/opp/all
num_feature = range(5, 51, 1)  # number of feature for training
num_fc = 6216  # number of functional connectivity pairs
num_str = 202  # number of structural features
num_fun = 336  # number of functional features
hidden_node_1 = 5

# load data set from csv file
df_srpbs = pd.read_csv("/src/top_features_srpbs_all.csv")
df_opp = pd.read_csv("/src/top_features_opp_all.csv")
df_yale = pd.read_csv("/src/top_features_yale_all.csv")# setting list of features
feature_names = df_srpbs.columns.values.tolist()
feature_names.pop(0)

# data selection
mri_srpbs, df_fi, feature_names = cl.select_data(feature, df_srpbs, feature_names, num_fc, num_str)
mri_opp = cl.select_data(feature, df_opp, feature_names, num_fc, num_str)[0]
mri_yale = cl.select_data(feature, df_yale, feature_names, num_fc, num_str)[0]

# Loading columns set from features importance
df_fi = cl.sort_features(df_fi, algo_sort)


# defining data array
class DataList:
    def __init__(self, input_df):
        # Create label array
        self.label = torch.FloatTensor(input_df.to_numpy()[:, 0])
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

# Sorting the data frame according to importance
df_srpbs = df_srpbs[df_fi.columns]
df_opp = df_opp[df_fi.columns]
df_yale = df_yale[df_fi.columns]

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

    # X_train, X_test, y_train, y_test = train_test_split(mri_data, mri_label, test_size=0.3, random_state=69)
    # print("X_train_size:", X_train.shape)
    # print("Y_train_size:", y_train.shape)
    # train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    # test_data = TestData(torch.FloatTensor(X_test))
    # BATCH_SIZE = 88
    # train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=1)

    # # construct network
    # model = network(300, 200, 120, 1)
    # print(model)
    # EPOCHS = 150
    # learning_rate = 1e-1
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model.train()
    # for e in range(1, EPOCHS + 1):
    #     epoch_loss = 0
    #     epoch_acc = 0
    #     for X_batch, y_batch in train_loader:
    #         # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         optimizer.zero_grad()
    #         y_pred = model(X_batch)
    #         loss = criterion(y_pred, y_batch.unsqueeze(1))
    #         acc = binary_acc(y_pred, y_batch.unsqueeze(1))
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #         epoch_acc += acc.item()
    #     print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
    #
    # # accuracy
    # y_pred_list = []
    # model.eval()
    # with torch.no_grad():
    #     for X_batch in test_loader:
    #         y_test_pred = model(X_batch)
    #         y_test_pred = torch.sigmoid(y_test_pred)
    #         y_pred_tag = torch.round(y_test_pred)
    #         y_pred_list.append(y_pred_tag)
    #
    # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    #
    # print(confusion_matrix(y_test, y_pred_list))
    # print(classification_report(y_test, y_pred_list))

    # k-fold validation

    kfold = KFold(n_splits=10, random_state=1, shuffle=True)

    # clearing list for each fold loop
    def clear_list(x):
        x.acc_pfold.clear()
        x.prc_pfold.clear()
        x.rec_pfold.clear()


    clear_list(srpbs)
    clear_list(opp)
    clear_list(yale)

    for train, test in kfold.split(srpbs_data, srpbs.label):
        # construct network
        scaler = MinMaxScaler()
        scaler.fit(srpbs_data[train])
        X_train = torch.FloatTensor(scaler.transform(srpbs_data[train]))
        model = network(i, hidden_node_1, 1)
        EPOCHS = 150
        learning_rate = 1e-1
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        for e in range(1, EPOCHS + 1):
            epoch_loss = 0
            epoch_acc = 0
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, srpbs.label[train].unsqueeze(1))
            acc = binary_acc(y_pred, srpbs.label[train].unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # generate generalization metrics
        model.eval()

        def app_fold_result(y_data, y_label, x):
            with torch.no_grad():
                X_test = torch.FloatTensor(scaler.transform(y_data))
                y_test_pred = model(X_test)
                y_test_pred = torch.round(torch.sigmoid(y_test_pred))
            x.acc_pfold.append(metrics.accuracy_score(y_label, y_test_pred) * 100)
            x.prc_pfold.append(metrics.precision_score(y_label, y_test_pred) * 100)
            x.rec_pfold.append(metrics.recall_score(y_label, y_test_pred) * 100)


        if dataset == "self":
            app_fold_result(srpbs_data[test], srpbs_data[test], srpbs)
        elif dataset == "opp":
            app_fold_result(opp_data, opp.label, opp)
        elif dataset == "yale":
            app_fold_result(yale_data, yale.label, yale)
        else:
            app_fold_result(srpbs_data[test], srpbs.label[test], srpbs)
            app_fold_result(opp_data, opp.label, opp)
            app_fold_result(yale_data, yale.label, yale)

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
    else:
        print('------------------------------------------------------------------------')
        app_test_result(srpbs)
        app_test_result(opp)
        app_test_result(yale)
    # print('------------------------------------------------------------------------')
    # print(f'Score per fold w/ total fold number = {k}')
    # for j in range(0, len(acc_pfold)):
    #     print('------------------------------------------------------------------------')
    #     print(f'> Fold {j + 1} - Accuracy: {acc_pfold[j]}% - Precision: {prc_pfold[j]}% - Recall: {rec_pfold[j]}%')
    # print('------------------------------------------------------------------------')
    # print(f'Training for k-fold number = {k} ...')
    # print(f'> Accuracy: {np.mean(acc_pfold)}% (+- {np.std(acc_pfold)}%)')
    # print(f'> Precision: {np.mean(prc_pfold)}% (+- {np.std(prc_pfold)}%)')
    # print(f'> Recall: {np.mean(rec_pfold)}% (+- {np.std(rec_pfold)}%)')


# provide average scores
def print_score(x, y, z):
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
            f'Recall: {z.rec_folds[j]}%')


print_score(srpbs, opp, yale)


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


def plot_score(x, y, z):
    plt.plot(num_feature, x.acc_mean_folds, color='red', marker='.', label="jas")
    plt.plot(num_feature, y.acc_mean_folds, color='blue', marker='x', label="opp")
    plt.plot(num_feature, z.acc_mean_folds, color='green', marker='*', label="yale")
    # plt.plot(num_feature, x.prc_folds, color='blue', marker='x', label="precision")
    # plt.plot(num_feature, x.rec_folds, color='green', marker='s', label="recall")
    annot_max(num_feature, y.acc_mean_folds)
    plt.title(f"Accuracy Analysis for Number of Features in ANN ({feature}/{algo_sort})")
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()


plot_score(srpbs, opp, yale)
