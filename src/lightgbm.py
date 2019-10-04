import os
import pandas as pd
import gc
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb


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

train = pd.read_pickle(os.path.join(folder_path, 'train3.pkl'))
test = pd.read_pickle(os.path.join(folder_path, 'test3.pkl'))
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
# ----------------------------------------- ENCODING -----------------------------------------------
for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

# ---------- Useful features derived by Permutation

useful_cols = ['addr2_C5count','V258','V257','V156','V149','addr2_C5count_oh_9','C7','V201','V189','ProductCD_oh_3',
                'V175_oh_1','C1_count','V154','C7_count','C14','C1','V94_oh_1','V251_oh_3','V62','M4_oh_3','V244',
                'V223','id_35_oh_1','C12','V62_te','V53_oh_3','V153','V45','addr2_count','id_17','V12_oh_3','M5_oh_2',
                'C4','V17_oh_1','C13','ProductCD_count_oh_0','V262','V53_oh_0','V246','V197_oh_1','V12','V154_oh_1',
                'V283','V154_te','ProductCD_count_oh_2','V294','V58_oh_2','V12_te','V223_oh_1','V53_te','V187',
                'id_32_oh_5','V175','V295','C12_count','card6','V317','C14_count','C13_count','V242','V200','V188',
                'V20_oh_1','V147','V197','V281','V148','V322','V296','V53','V79_oh_4','id_34_oh_3','card6_count',
                'V142_te','card6_te','ProductCD','C10','C2','C11_count','V142_oh_1','V332','V197_te','V82_oh_2',
                'V153_oh_1','V224','id_11','card6_oh_2','V61_te','C8_count','R_emaildomain_1','V94','V194_oh_1',
                'V318','V243','ProductCD_oh_2','V108_te','P_emaildomain_2_oh_0','dist2_nan_oh_1','V259','V195_oh_2',
                'V333','M5','ProductCD_oh_1','V261','DeviceType_oh_1','M4','card3','V63_oh_1','ProductCD_count_oh_1',
                'V223_te','card1_amtcount','V194','V82','addr1_C5count','V184','V74_oh_2','V190','V35','D13','C10_count',
                'V170','V85_oh_1','V245','V70_te','V79','P_emaildomain_1','id_36_oh_1','V123','V174','V146','id_22',
                'V251_oh_2','V114_oh_2','id_13','V112','V157','V331','V232','C2_count','V239','V251_te','V143','id_32_oh_4',
                'V78','V171','V63_oh_2','UserID_base','V255','V58','V73_oh_2','V217','V173_oh_1','V83_te','V178',
                'V83_oh_2','addr1','V54','id_01','V251_oh_1','V153_te','card2_amtcount','V327_te','id_38_oh_2',
                'V313','V56','V36','Transaction_day_of_week_oh_4','V174_oh_2','V250_oh_2','V194_oh_3','TransactionAmt',
                'id_34_oh_2','V315','V72_oh_3','V248','M4_oh_0','UserID_mid','V180','UserID_full','V161','V330',
                'M5_te','V326','V34','V250_te','addr1_count','V69_oh_1','V167','TransactionAmt_id_base_std','V328',
                'card2_C5count','V76_oh_3','V99','id_15','V229','V195_oh_3','V308','V88','V218','V336','V124_oh_2',
                'V282','V62_oh_3','V191','V61_oh_2','V309','V250_oh_1','id_18','V173','addr2','id_37','V85_oh_2',
                'TransactionAmt_mod1','V220','V7','D15','id_05','V279','id_04_te','id_09','card1','V192','V306',
                'V17','V288','card5','id_33_0','V136','dist2','V286_oh_2','TransactionAmt_id_mid_mean','card4_te',
                'C5','V181','V125_oh_3','V97','V288_oh_1','V104_te','M2','V280','V60','Transaction_hour','D10',
                'card4','TransactionAmt_id_full_std','V338','V123_oh_3','V83_oh_3','V125','V292','V9_te','V104',
                'V311','V46_te','D5','V288_oh_0','V297_oh_8','V297_oh_10','V289_oh_5','V288_oh_5','FirstTransactionFlag_te',
                'id_37_te','dist1','id_35_te','id_34_te','V241_te','id_16_te','id_12_te','V328_te','V305_te','V304_te',
                'V302_te','V300_te','V297_te','V286_te','M9_te','V33_te','M8_te','M7_te','M6_te','id_04_oh_12','id_04_oh_13',
                'id_04_oh_14','V121','id_18_oh_16','id_04_oh_15','id_18_oh_3','id_18_oh_2','id_18_oh_1','V122']

useful_cols.append('TransactionID')
useful_cols.append('TransactionDT')
test = test[useful_cols]
useful_cols.append('isFraud')
train = train[useful_cols]


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- DEV/VAL SPLIT ------------------------------------------

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

# --------------------------------------------------------------------------------------------------
# ----------------------------------------- IMPUTATION ---------------------------------------------
X_dev = X_dev.fillna(-999)
X_val = X_val.fillna(-999)

X_dev = X_dev.replace([np.inf, -np.inf], -999)
X_dev[X_dev.isnull()] = '-999'

# --------------------------------------------------------------------------------------------------
# ----------------------------------------- LGB -----------------------------------------------


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


params = {
                    'objective': 'binary','boosting_type': 'gbdt','metric': 'auc',
                    'n_jobs': -1,
                    'reg_lambda': 0.2,
                    'reg_alpha': 0.4,

                    'learning_rate': 0.01,
                    'num_leaves': 800,
                    'min_data_in_leaf': 120,

                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.5,

                    'n_estimators': cb_iters,
                    'verbose': -1,
                    'seed': random_state,
                    'early_stopping_rounds': early_stopping_rounds
                }

lgtrain = lgb.Dataset(X_dev, label=y_dev)
lgval = lgb.Dataset(X_val, label=y_val)
evals_result = {}
model = lgb.train(params, lgtrain, cb_iters, valid_sets=[lgval], early_stopping_rounds=early_stopping_rounds,
                  verbose_eval=200, evals_result=evals_result)

print("-"*30 + 'VALIDATION' + "-"*30)
print(model.best_score['valid_0'][params['metric']])
print("-"*70)
val_score = np.round(model.best_score['valid_0'][params['metric']], 5)

if DoForecast:
    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- FULL LGB -----------------------------------------------
    params['n_estimators'] = model.best_iteration
    params['early_stopping_rounds'] = None

    y_dev = train['isFraud']
    X_dev = train.drop(columns='isFraud')
    del X_dev['TransactionDT']
    del X_dev['TransactionID']
    lgtrain = lgb.Dataset(X_dev, label=y_dev)

    model = lgb.train(params, lgtrain, model.best_iteration, verbose_eval=200)

    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- PREDICTION ---------------------------------------------

    del test['TransactionDT']
    ids = test['TransactionID']
    del test['TransactionID']

    print(X_dev.shape, test.shape)

    preds = model.predict(test)
    y_preds = pd.DataFrame(columns=['TransactionID'])
    y_preds['TransactionID'] = ids
    y_preds['isFraud'] = preds

    print("Saving submission file")
    script_name = os.path.basename(__file__).split('.')[0]
    y_preds[['TransactionID', 'isFraud']].to_csv('{}__{}.csv'.format(script_name, str(val_score)), index=False)
