import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from dateutil.relativedelta import relativedelta
from xgboost import XGBClassifier

# -------------------------------------------------------------------------------------------------
# ------------------------------------------------- INPUTS ----------------------------------------
random_state = 42
nrows = None

np.random.seed(random_state)
cb_iters = 10000
early_stopping_rounds = 250
folder_path = r'../input'

TEST_FLAG = False
DoForecast = True

print("cb_iters:{0}\nTEST_FLAG: {1}\nDoForecast: {2}\n".format(cb_iters, TEST_FLAG, DoForecast))
# --------------------------------------------------------------------------------------------------
# ------------------------------------------------- LOADING DATA -----------------------------------

train = pd.read_pickle(os.path.join(folder_path, 'train2.pkl'))
test = pd.read_pickle(os.path.join(folder_path, 'test2.pkl'))
print("Train size is {0}, test size is {1}".format(train.shape[0], test.shape[0]))

cols_to_drop = []
categorical_features = []

if TEST_FLAG:
    train = train[train['TransactionDT'] <= 60 * 60 * 24 * 90] #Only 90 days for testing
    test = test[test['TransactionDT'] <= 60 * 60 * 24 * 270]
# ----- Function for Dev/Val split by time

def dev_val_split(df, lag_days=10, val_days=20):
    max_dt = df['TransactionDT'].max()
    val_start = max_dt - (val_days * 60 * 60 * 24)
    dev_end = max_dt - (val_days * 60 * 60 * 24 + lag_days * 60 * 60 * 24)

    df_dev = df[df['TransactionDT'] <= dev_end]
    df_val = df[df['TransactionDT'] >= val_start]

    return df_dev, df_val


print(train.shape, test.shape)

# --------------------------------------------------------------------------------------------------
# ------------------------------------------------- FEATURES ---------------------------------------

# ----------- DoW and Hour
train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

cols_to_drop.append("TransactionDT")
categorical_features = categorical_features + ["Transaction_day_of_week", "Transaction_hour"]


# ---------- dist1/dist2
train['dist1_nan'] = np.where(np.isnan(train['dist1']), 1, 0)
train['dist2_nan'] = np.where(np.isnan(train['dist2']), 1, 0)
test['dist1_nan'] = np.where(np.isnan(test['dist1']), 1, 0)
test['dist2_nan'] = np.where(np.isnan(test['dist2']), 1, 0)

# cols_to_drop = cols_to_drop + ['dist1', 'dist2']


# ----------- P_email/R_email

train['P_emaildomain_1'] = train['P_emaildomain'].str.split(".", expand=True)[0].astype(str)
train['R_emaildomain_1'] = train['R_emaildomain'].str.split(".", expand=True)[0].astype(str)
train['P_emaildomain_2'] = train['P_emaildomain'].str.split(".", expand=True)[1].astype(str)
train['R_emaildomain_2'] = train['R_emaildomain'].str.split(".", expand=True)[1].astype(str)

test['P_emaildomain_1'] = test['P_emaildomain'].str.split(".", expand=True)[0].astype(str)
test['R_emaildomain_1'] = test['R_emaildomain'].str.split(".", expand=True)[0].astype(str)
test['P_emaildomain_2'] = test['P_emaildomain'].str.split(".", expand=True)[1].astype(str)
test['R_emaildomain_2'] = test['R_emaildomain'].str.split(".", expand=True)[1].astype(str)

cols_to_drop = cols_to_drop + ['P_emaildomain', 'R_emaildomain']

# ----------- Impute card
i_cols = ['TransactionID', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6']

full_df = pd.concat([train[i_cols], test[i_cols]])


i_cols = ['card2', 'card3', 'card4', 'card5', 'card6']

for col in i_cols:
    temp_df = full_df.groupby(['card1', col])[col].agg(['count']).reset_index()
    temp_df = temp_df.sort_values(by=['card1', 'count'], ascending=False).reset_index(drop=True)
    del temp_df['count']
    temp_df = temp_df.drop_duplicates(keep='first').reset_index(drop=True)
    temp_df.index = temp_df['card1'].values
    temp_df = temp_df[col].to_dict()
    full_df[col] = np.where(full_df[col].isna(), full_df['card1'].map(temp_df), full_df[col])

i_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
for col in i_cols:
    train[col] = full_df[full_df['TransactionID'].isin(train['TransactionID'])][col].values
    test[col] = full_df[full_df['TransactionID'].isin(test['TransactionID'])][col].values


# ---------- UserID

def get_user_id(df_all, by, col_name, start_point=datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')):

    # by = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction']

    df = df_all.copy()
    df['TransactionDays'] = np.round(df['TransactionDT']/(60*60*24), 0)
    df['TransactionDate'] = [start_point + relativedelta(seconds=secs) for secs in df['TransactionDT']]
    df['D1'] = np.where(np.isnan(df['D1']), 0, df['D1'])

    df['FirstTransaction'] = [dt1-relativedelta(days=days) for dt1, days in zip(df['TransactionDate'], df['D1'])]
    df['FirstTransaction'] = [datetime.datetime.strftime(d, '%Y-%m-%d') for d in df['FirstTransaction']]
    df['D3'] = np.where(np.isnan(df['D3']), 0, df['D3'])

    df[by] = df[by].fillna(-99)

    grouped = df.groupby(by, as_index=False)['TransactionID'].min()
    grouped = grouped.rename(columns={'TransactionID': col_name})

    df = pd.merge(df, grouped, on=by, how='left')

    df = df.sort_values(['TransactionDays'], ascending=True).groupby([col_name]).head(df.shape[0])
    df['diffs'] = df.sort_values(['TransactionDT'], ascending=True).groupby([col_name])['TransactionDT'].transform(lambda x: x.diff())

    df['firstInRow'] = np.where(np.isnan(df['diffs']), 1, 0)
    df['diffs'] = np.where(np.isnan(df['diffs']), 0, df['diffs'])
    df['diffs'] = np.round(df['diffs'] / (60 * 60 * 24), 0)

    return df

# Combine Train and Test
train['FirstTransactionFlag'] = np.where((train['D1'] == 0) & (train['D3']==0), 1, 0)
test['FirstTransactionFlag'] = np.where((test['D1'] == 0) & (test['D3']==0), 1, 0)

train['isTest'] = 0
test['isTest'] = 1
test.insert(1, "isFraud", 0)

df_all = pd.concat([train, test], axis=0)

# ----getting userid
by = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction']
df = get_user_id(df_all, by, 'UserID')

train = df[df['isTest'] == 0].sort_values(['TransactionID'])
test = df[df['isTest'] == 1].sort_values(['TransactionID'])
del df_all, df
print(train.shape, test.shape)


cols_to_drop = cols_to_drop + ['TransactionDays', 'TransactionDate', 'diffs', 'isTest', 'FirstTransaction']
test = test.drop(['isFraud'], axis=1)
categorical_features.append("UserID")

# ------------------ ID 2
# Combine Train and Test

train['isTest'] = 0
test['isTest'] = 1
test.insert(1, "isFraud", 0)
df_all = pd.concat([train, test], axis=0)

# ----getting userid
by = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'FirstTransaction',
         'V29', 'V53', 'V55', 'V57', 'V59', 'V61', 'V63', 'V69', 'V138', 'V139', 'V140', 'V141', 'V142',
         'V144', 'V145', 'V146', 'V147', 'V148', 'V150', 'V151', 'V152', 'V157', 'V159', 'V160',
         'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V305', 'V311', 'V322', 'V323',
         'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330',
         'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339']

df = get_user_id(df_all, by, 'UserID2')

train = df[df['isTest'] == 0].sort_values(['TransactionID'])
test = df[df['isTest'] == 1].sort_values(['TransactionID'])
del df_all, df


test = test.drop(['isFraud'], axis=1)
categorical_features.append("UserID2")

cols_to_drop = cols_to_drop + ['D2', 'M1', 'V1', 'V2', 'V14', 'V15', 'V16', 'V18', 'V21', 'V22', 'V23', 'V24', 'V25',
                               'V26', 'V27', 'V28', 'V31', 'V32', 'V39', 'V41', 'V42', 'V43', 'V50', 'V55', 'V57',
                               'V65', 'V66', 'V67', 'V68',
                                'V159', 'V160', 'V164', 'V165', 'V166', 'V202', 'V203',
                                'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210',
                                'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V263',
                                'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270',
                                'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278'
                               ]

# -------- Counts and amtcounts
# -- from https://www.kaggle.com/whitebird/a-method-to-valid-offline-lb-9506

cols = "TransactionDT,TransactionAmt,ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,M1,M2,M3,M4,M5,M6,M7,M8,M9".split(",")
train_test = train[cols].append(test[cols])

for col in "ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14".split(","):
    col_count = train_test.groupby(col)['TransactionDT'].count()
    train[col + '_count'] = train[col].map(col_count)
    test[col + '_count'] = test[col].map(col_count)

for col in "card1,card2,card5,addr1,addr2".split(","):
    col_count = train_test.groupby(col)['TransactionAmt'].mean()
    train[col + '_amtcount'] = train[col].map(col_count)
    test[col + '_amtcount'] = test[col].map(col_count)

    col_count1 = train_test[train_test['C5'] == 0].groupby(col)['C5'].count()
    col_count2 = train_test[train_test['C5'] != 0].groupby(col)['C5'].count()

    train[col + '_C5count'] = train[col].map(col_count2) / (train[col].map(col_count1) + 0.01)
    test[col + '_C5count'] = test[col].map(col_count2) / (test[col].map(col_count1) + 0.01)

# ----- Cents and different pkl

train['cents'] = np.round(train['TransactionAmt'] - np.floor(train['TransactionAmt']), 2)
test['cents'] = np.round(test['TransactionAmt'] - np.floor(test['TransactionAmt']), 2)

# ------ Grouped by ID
cols = ['TransactionID', 'TransactionAmt', 'UserID']
train_test = train[cols].append(test[cols])
col = 'UserID'
train['TransactionAmt_id_mean'] = train['TransactionAmt']/train_test.groupby(col)['TransactionAmt'].transform('mean')[0:train.shape[0]]
test['TransactionAmt_id_mean'] = test['TransactionAmt']/train_test.groupby(col)['TransactionAmt'].transform('mean')[train.shape[0]:]

train['TransactionAmt_id_std'] = train['TransactionAmt']/train_test.groupby(col)['TransactionAmt'].transform('std')[0:train.shape[0]]
test['TransactionAmt_id_std'] = test['TransactionAmt']/train_test.groupby(col)['TransactionAmt'].transform('std')[train.shape[0]:]

# --------------------------------------------------------------------------------------------------
# ------------------------------------------------- ENCODING ---------------------------------------

cols_to_drop = list(set(cols_to_drop).intersection(set(train.columns)))
cols_to_drop.remove('TransactionDT')
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)


for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

train = train.fillna(-999)
test = test.fillna(-999)


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- DATA FOR XGB (HoldOut) ----------------------------------

df_dev, df_val = dev_val_split(train)
del df_dev['TransactionDT']
del df_val['TransactionDT']
del df_dev['TransactionID']
del df_val['TransactionID']

y_dev = df_dev['isFraud']
X_dev = df_dev.drop(columns='isFraud')
y_val = df_val['isFraud']
X_val = df_val.drop(columns='isFraud')

print("X_dev shape is: ")
print(X_dev.shape)


# ----- Categorical Features for CB
categorical_features = categorical_features + ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
categorical_features = categorical_features + ['M'+str(i) for i in range(1, 10)]
categorical_features = categorical_features + ['id_'+str(i) for i in range(12, 39)]
categorical_features = categorical_features + ['DeviceType', 'DeviceInfo', 'id_04']
categorical_features = categorical_features + ['device_name', 'device_version', 'OS_id_30', 'version_id_30',
                                                'browser_id_31', 'version_id_31', 'screen_width', 'screen_height',
                                                'AmtCentsLen', 'Transaction_day_of_week', 'Transaction_hour', 'isFirst']

categorical_features = list(set(categorical_features).intersection(set(X_dev.columns)))


params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'learning_rate': 0.01,
        'base_score': 0.5,

        'max_depth': 20,
        'min_child_weight': 1,
        'subsample': 0.85,
        'colsample_bytree': 0.5,
        'alpha': 4, 'gamma': 0.1,


        'missing': -999,
        'n_estimators': cb_iters,
        'early_stopping_rounds': early_stopping_rounds,
        'n_jobs': 32,
        'random_state': 21,

        'seed': 21,
        'silent': None,
        'verbosity': 1
}

model = XGBClassifier(**params)
evals_result = {}
model.fit(X_dev, y_dev, eval_metric="auc", eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)


print("-"*30 + 'VALIDATION' + "-"*30)
print(model.best_score)
print("-"*70)
val_score = np.round(model.best_score, 5)

if DoForecast:
    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- FULL XGB ------------------------------------------------
    params['n_estimators'] = model.best_iteration

    y_dev = train['isFraud']
    X_dev = train.drop(columns='isFraud')
    del X_dev['TransactionDT']
    del X_dev['TransactionID']

    model = XGBClassifier(**params)
    model.fit(X_dev, y_dev)

    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- PREDICTION ---------------------------------------------

    del test['TransactionDT']
    ids = test['TransactionID']
    del test['TransactionID']

    print(X_dev.shape, test.shape)

    preds = model.predict_proba(test, ntree_limit=params['n_estimators'])[:, 1]

    y_preds = pd.DataFrame(columns=['TransactionID'])
    y_preds['TransactionID'] = ids
    y_preds['isFraud'] = preds

    print("Saving submission file")
    script_name = os.path.basename(__file__).split('.')[0]
    y_preds[['TransactionID', 'isFraud']].to_csv('{}__{}.csv'.format(script_name, str(val_score)), index=False)

