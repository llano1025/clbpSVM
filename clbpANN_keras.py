import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load data set from csv file
df_mri = pd.read_csv("F:/bp_t/top_features.csv")
df_opp = pd.read_csv("F:/bp_t/top_features_opp.csv")

# setting list of features
feature_names = df_mri.columns.values.tolist()
feature_names.pop(0)
print(feature_names)

# setting data array and split
mri_arr = df_mri.to_numpy()
mri_data = mri_arr[:, 1:]
mri_label = mri_arr[:, 0]
opp_arr = df_opp.to_numpy()
opp_data = opp_arr[:, 1:]
opp_label = opp_arr[:, 0]
# X_train, X_test, y_train, y_test = train_test_split(mri_data, mri_label, test_size=0.3, random_state=69)
# print("X_train_size:", X_train.shape)
# print("Y_train_size:", y_train.shape)
#
# # define the keras model
# model = Sequential()
# model.add(Dense(200, input_dim=300, activation='relu'))
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid'))
#
# # compile the keras model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # fit the keras model on the dataset
# model.fit(X_train, y_train, epochs=150, batch_size=len(y_train))
#
# # evaluate the keras model
# _, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))
#
# y_pred = model.predict(X_test)
# y_pred = y_pred.round()
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # leave one out validation
# loo = LeaveOneOut()
# loo.get_n_splits(mri_data)
# y_true, y_pred = list(), list()
# for train_index, test_index in loo.split(mri_data):
#     X_train, X_test = mri_data[train_index, :], mri_data[test_index, :]
#     y_train, y_test = mri_label[train_index], mri_label[test_index]
#     model.fit(X_train, y_train, epochs=150, batch_size=len(y_train))
#     yhat = model.predict(X_test)
#     y_true.append(y_test[0])
#     y_pred.append(yhat[0].round())
#
# acc = accuracy_score(y_true, y_pred)
# print('Leave One Out Accuracy: %.3f' % acc)

# k-fold validation
acc_pfold, loss_pfold = list(), list()
acc_mean_folds, acc_max_folds, acc_min_folds, loss_folds = list(), list(), list(), list()
folds_no = range(2, 11, 1)

for k in folds_no:

    kfold = KFold(n_splits=k, random_state=1, shuffle=True)
    acc_pfold.clear()
    loss_pfold.clear()

    for train, test in kfold.split(mri_data, mri_label):
        # define model architecture
        model = Sequential()
        model.add(Dense(200, input_dim=300, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit data to model
        scaler = MinMaxScaler()
        scaler.fit(mri_data[train])
        X_train = scaler.transform(mri_data[train])
        model.fit(X_train, mri_label[train], epochs=150, batch_size=len(train), verbose=0)

        # Generate generalization metrics
        X_test = scaler.transform(mri_data[test])
        scores = model.evaluate(X_test, mri_label[test], verbose=0)
        # scores = model.evaluate(opp_data, opp_label, verbose=0)
        acc_pfold.append(scores[1] * 100)
        loss_pfold.append(scores[0])

    # provide average scores
    acc_mean_folds.append(np.mean(acc_pfold))
    acc_max_folds.append(np.max(acc_pfold)-np.mean(acc_pfold))
    acc_min_folds.append(np.mean(acc_pfold)-np.min(acc_pfold))
    loss_folds.append(np.mean(loss_pfold))
    print('------------------------------------------------------------------------')
    print(f'Score per fold w/ total fold number = {k}')
    for i in range(0, len(acc_pfold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_pfold[i]} - Accuracy: {acc_pfold[i]}%')
    print('------------------------------------------------------------------------')
    print(f'Training for k-fold number = {k} ...')
    print(f'> Accuracy: {np.mean(acc_pfold)} (+- {np.std(acc_pfold)})')
    print(f'> Loss: {np.mean(loss_pfold)}')

# provide average scores
print('------------------------------------------------------------------------')
print('Score per folds')
for i in range(0, len(acc_mean_folds)):
    print('------------------------------------------------------------------------')
    print(f'> No. of fold : {i + 2} - Loss: {loss_folds[i]} - Accuracy: {acc_mean_folds[i]}%')

# show the plot
plt.errorbar(folds_no, acc_mean_folds, yerr=[acc_min_folds, acc_max_folds], fmt='o', ecolor='r', color='b')
plt.title('Sensitivity Analysis for k number')
plt.xlabel('k-number')
plt.ylabel('Accuracy(%)')
plt.show()
