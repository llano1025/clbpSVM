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
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden1_size)
        # self.batchnorm2 = nn.BatchNorm1d(hidden2_size)

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


# general set up
# rfe (Recursive Feature Elimination) / etc (Extra Trees Classifier) / ust (Uni-variate Statistical Tests)
# pls (Partial Least Squares)/ xgb (eXtreme Gradient Boosting) / sbe (Sequential Backward Elimination)
algo_sort = "pls"  # Top features ranked by featureSelection.py
feature = "all"  # all/ str / fc / fun
dataset = "opp"  # self/opp
num_feature = 30
num_fc = 6216  # number of functional connectivity pairs
num_str = 202  # number of structural features
num_fun = 336  # number of functional features
num_folds = range(2, 21, 1)

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
mri_label = torch.FloatTensor(mri_label)
opp_label = torch.FloatTensor(opp_label)

df_mri = df_srpbs[df_fi.columns]
df_opp = df_opp[df_fi.columns]

# defining data array
mri_arr = df_mri.to_numpy()
opp_arr = df_opp.to_numpy()
mri_arr[np.isnan(mri_arr)] = 0
opp_arr[np.isnan(opp_arr)] = 0
mri_data = mri_arr[:, 0:num_feature]
opp_data = opp_arr[:, 0:num_feature]

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
acc_pfold, prc_pfold, rec_pfold = list(), list(), list()
acc_mean_folds, prc_folds, rec_folds = list(), list(), list()
acc_max_folds, acc_min_folds = list(), list()

for k in num_folds:

    kfold = KFold(n_splits=k, random_state=1, shuffle=True)
    acc_pfold.clear()
    prc_pfold.clear()
    rec_pfold.clear()

    for train, test in kfold.split(mri_data, mri_label):
        # construct network
        scaler = MinMaxScaler()
        scaler.fit(mri_data[train])
        X_train = torch.FloatTensor(scaler.transform(mri_data[train]))
        model = network(30, 10, 1)
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
            loss = criterion(y_pred, mri_label[train].unsqueeze(1))
            acc = binary_acc(y_pred, mri_label[train].unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # generate generalization metrics
        model.eval()
        if dataset == "self":
            with torch.no_grad():
                X_test = torch.FloatTensor(scaler.transform(mri_data[test]))
                y_test_pred = model(X_test)
                y_test_pred = torch.round(torch.sigmoid(y_test_pred))
            acc_pfold.append(metrics.accuracy_score(mri_label[test], y_test_pred) * 100)
            prc_pfold.append(metrics.precision_score(mri_label[test], y_test_pred) * 100)
            rec_pfold.append(metrics.recall_score(mri_label[test], y_test_pred) * 100)
        else:
            with torch.no_grad():
                X_test = torch.FloatTensor(scaler.transform(opp_data))
                y_test_pred = model(X_test)
                y_test_pred = torch.round(torch.sigmoid(y_test_pred))
            acc_pfold.append(metrics.accuracy_score(opp_label, y_test_pred) * 100)
            prc_pfold.append(metrics.precision_score(opp_label, y_test_pred) * 100)
            rec_pfold.append(metrics.recall_score(opp_label, y_test_pred) * 100)

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
        f'> No. of fold : {i + 2} - Accuracy: {acc_mean_folds[i]}% - Precision: {prc_folds[i]}% - Recall: {rec_folds[i]}%')

# show the plot
plt.errorbar(num_folds, acc_mean_folds, yerr=[acc_min_folds, acc_max_folds], fmt='o', ecolor='r', color='b')
plt.title(f"Sensitivity Analysis for k number in ANN ({feature}/{algo_sort})")
plt.xlabel('k-number')
plt.ylabel('Accuracy(%)')
plt.show()
