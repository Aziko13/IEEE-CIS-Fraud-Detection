import pandas as pd
import numpy as np


sub1 = pd.read_csv(r'path_to_catboost_solution')
sub2 = pd.read_csv(r'path_to_lightgbm_solution')
sub3 = pd.read_csv(r'path_to_xgboost_solution')
sub4 = pd.read_csv(r'LiteMort solution from https://github.com/closest-git/ieee_fraud')

sub_mean_file = "sub_mean.csv"

# ----------- Average of submissions
subs = [sub1, sub2, sub3, sub4]
# weights = [0.3, 0.2, 0.2, 0.3]
weights = None

# -------------------------------

sub_mean = pd.DataFrame()

for i in range(0, len(subs)):
    sub_i = subs[i].sort_values(['TransactionID'])
    if i == 0:
        sub_mean['TransactionID'] = sub_i['TransactionID']

    if weights is None:
        w = 1
    else:
        w = weights[i]

    sub_mean['isFraud_'+str(i)] = sub_i['isFraud'] * w


sub_mean['isFraud'] = np.mean(sub_mean.iloc[:, 1:], axis=1)
sub_mean[['TransactionID', 'isFraud']].to_csv(sub_mean_file, index=False)

