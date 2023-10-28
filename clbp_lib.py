import pandas as pd


def select_data(feature, input_df, feature_names, num_fc, num_str):
    if feature == "all":
        output_df = input_df.iloc[:, 1:]
        select_fi = pd.read_csv("/fs/features_importance.csv")
    elif feature == "fc":
        output_df = input_df.iloc[:, 1:num_fc]
        feature_names = feature_names[:num_fc - 1]
        select_fi = pd.read_csv("/fs/features_importance_fc.csv")
    elif feature == "str":
        output_df = input_df.iloc[:, num_fc + 1:num_fc + num_str + 1]
        feature_names = feature_names[num_fc:num_fc + num_str]
        select_fi = pd.read_csv("/fs/features_importance_str.csv")
    else:
        output_df = input_df.iloc[:, num_fc + num_str + 2:]
        feature_names = feature_names[num_fc:num_fc + num_str + 1:]
        select_fi = pd.read_csv("/fs/features_importance_fun.csv")
    return output_df, select_fi, feature_names


def sort_features(df_fi, algo_sort):
    df_fi.set_index('feature', inplace=True)
    if algo_sort == "pls":
        df_fi = df_fi.sort_values(by='pls', axis=1, ascending=True)
    elif algo_sort == "rfe":
        df_fi = df_fi.sort_values(by='rfe', axis=1, ascending=True)
    elif algo_sort == "etc":
        df_fi = df_fi.sort_values(by='etc', axis=1, ascending=True)
    elif algo_sort == "ust":
        df_fi = df_fi.sort_values(by='ust', axis=1, ascending=True)
    elif algo_sort == "xgb":
        df_fi = df_fi.sort_values(by='xgb', axis=1, ascending=True)
    elif algo_sort == "sbe":
        df_fi = df_fi.sort_values(by='sbe', axis=1, ascending=True)
    elif algo_sort == "pls_hku1":
        df_fi = df_fi.sort_values(by='pls_hku1', axis=1, ascending=True)
    elif algo_sort == "pls_hku2":
        df_fi = df_fi.sort_values(by='pls_hku2', axis=1, ascending=True)
    elif algo_sort == "pls_hku3":
        df_fi = df_fi.sort_values(by='pls_hku3', axis=1, ascending=True)
    elif algo_sort == "pls_srpbs":
        df_fi = df_fi.sort_values(by='pls_srpbs', axis=1, ascending=True)
    elif algo_sort == "pls_opp":
        df_fi = df_fi.sort_values(by='pls_opp', axis=1, ascending=True)
    elif algo_sort == "pls_yale":
        df_fi = df_fi.sort_values(by='pls_yale', axis=1, ascending=True)
    else:  # pls_cambridge
        df_fi = df_fi.sort_values(by='pls_cambridge', axis=1, ascending=True)
    return df_fi

