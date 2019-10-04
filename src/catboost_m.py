import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# -------------------------------------------------------------------------------------------------
# ------------------------------------------------- INPUTS ----------------------------------------
random_state = 42
nrows = None

np.random.seed(random_state)
cb_iters = 10000
early_stopping_rounds = 250
folder_path = r''

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

# ---------- Useful features derived by RandomForest

useful_cols = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',
               'dist1', 'dist2', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
               'D1', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
               'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V7', 'V12', 'V13', 'V19', 'V20', 'V34', 'V35',
               'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V51', 'V52', 'V53', 'V54', 'V56', 'V59', 'V61',
               'V62', 'V64', 'V69', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V81', 'V82', 'V83', 'V86', 'V87',
               'V95', 'V96', 'V97', 'V99', 'V100', 'V101', 'V102', 'V103', 'V105', 'V123', 'V124', 'V125', 'V126',
               'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V140', 'V147',
               'V148', 'V149', 'V150', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V167', 'V168', 'V170',
               'V171', 'V172', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V182', 'V186', 'V187', 'V188',
               'V189', 'V190', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200', 'V201', 'V217',
               'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V227', 'V228', 'V229', 'V230', 'V231',
               'V232', 'V233', 'V234', 'V236', 'V237', 'V238', 'V239', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247',
               'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260',
               'V261', 'V262', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V287', 'V288', 'V289', 'V290',
               'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V298', 'V299', 'V301', 'V306', 'V307', 'V308', 'V309',
               'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321', 'V322',
               'V323', 'V324', 'V331', 'V332', 'V333', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09', 'id_11',
               'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32',
               'id_33', 'id_38', 'DeviceType', 'DeviceInfo', 'id_33_0', 'id_33_1', 'Transaction_day_of_week',
               'Transaction_hour', 'dist2_nan', 'P_emaildomain_1', 'R_emaildomain_1', 'P_emaildomain_2', 'UserID_base',
               'firstInRow', 'UserID_full', 'UserID_mid', 'ProductCD_count', 'card1_count', 'card2_count',
               'card3_count', 'card4_count', 'card5_count', 'card6_count', 'addr1_count', 'C1_count', 'C2_count',
               'C4_count', 'C5_count', 'C6_count', 'C7_count', 'C8_count', 'C9_count', 'C10_count', 'C11_count',
               'C12_count', 'C13_count', 'C14_count', 'card1_amtcount', 'card1_C5count', 'card2_amtcount',
               'card2_C5count', 'card5_amtcount', 'card5_C5count', 'addr1_amtcount', 'addr1_C5count', 'addr2_amtcount',
               'addr2_C5count', 'cents', 'TransactionAmt_id_base_mean', 'TransactionAmt_id_base_std',
               'TransactionAmt_id_mid_mean', 'TransactionAmt_id_mid_std', 'TransactionAmt_id_full_mean',
               'TransactionAmt_id_full_std', 'ProductCD_oh_1', 'card4_oh_3', 'card4_oh_4', 'card6_oh_2',
               'card6_oh_3', 'M2_oh_0', 'M2_oh_1', 'M3_oh_0', 'M3_oh_1', 'M4_oh_0', 'M4_oh_1', 'M4_oh_2',
               'M5_oh_0', 'M5_oh_1', 'M6_oh_0', 'M6_oh_1', 'M7_oh_0', 'M7_oh_2', 'M8_oh_0', 'M8_oh_2', 'M9_oh_1',
               'M9_oh_2', 'V12_oh_1', 'V12_oh_2', 'V12_oh_3', 'V13_oh_1', 'V13_oh_2', 'V19_oh_1', 'V19_oh_2',
               'V20_oh_1', 'V20_oh_2', 'V35_oh_1', 'V35_oh_2', 'V36_oh_1', 'V36_oh_2', 'V47_oh_2', 'V47_oh_3',
               'V53_oh_1', 'V53_oh_2', 'V53_oh_3', 'V54_oh_1', 'V54_oh_2', 'V61_oh_1', 'V61_oh_2', 'V61_oh_3',
               'V62_oh_1', 'V62_oh_2', 'V62_oh_3', 'V62_oh_4', 'V75_oh_1', 'V75_oh_2', 'V75_oh_3', 'V76_oh_1',
               'V76_oh_2', 'V82_oh_1', 'V82_oh_2', 'V82_oh_3', 'V83_oh_1', 'V83_oh_2', 'V83_oh_3', 'V123_oh_2',
               'V124_oh_2', 'V125_oh_2', 'V174_oh_1', 'V175_oh_1', 'V197_oh_3', 'V223_oh_1', 'V288_oh_1', 'V288_oh_2',
               'V289_oh_1', 'V289_oh_2', 'V289_oh_3', 'id_18_oh_0', 'id_18_oh_4', 'id_18_oh_6', 'id_38_oh_1',
               'id_38_oh_2', 'DeviceType_oh_0', 'DeviceType_oh_1', 'Transaction_day_of_week_oh_0',
               'Transaction_day_of_week_oh_1', 'Transaction_day_of_week_oh_2', 'Transaction_day_of_week_oh_3',
               'Transaction_day_of_week_oh_4', 'Transaction_day_of_week_oh_5', 'Transaction_day_of_week_oh_6',
               'dist2_nan_oh_0', 'dist2_nan_oh_1', 'P_emaildomain_2_oh_0', 'P_emaildomain_2_oh_2', 'firstInRow_oh_0',
               'firstInRow_oh_1', 'ProductCD_count_oh_1', 'ProductCD_count_oh_3', 'card4_count_oh_3',
               'card4_count_oh_4', 'card6_count_oh_3', 'card6_count_oh_4', 'addr2_C5count_oh_9', 'ProductCD_te',
               'card4_te', 'card6_te', 'M2_te', 'M3_te', 'M4_te', 'M5_te', 'M6_te', 'M7_te', 'M8_te', 'M9_te', 'V3_te',
               'V4_te', 'V5_te', 'V12_te', 'V13_te', 'V19_te', 'V20_te', 'V34_te', 'V35_te', 'V36_te', 'V40_te',
               'V46_te', 'V47_te', 'V48_te', 'V51_te', 'V52_te', 'V53_te', 'V54_te', 'V61_te', 'V62_te', 'V69_te',
               'V70_te', 'V75_te', 'V76_te', 'V79_te', 'V82_te', 'V83_te', 'V94_te', 'V123_te', 'V124_te', 'V125_te',
               'V153_te', 'V154_te', 'V173_te', 'V175_te', 'V194_te', 'V195_te', 'V197_te', 'V223_te', 'V247_te',
               'V250_te', 'V251_te', 'V260_te', 'V284_te', 'V288_te', 'V289_te', 'id_04_te', 'id_15_te', 'id_16_te',
               'id_18_te', 'id_32_te', 'id_38_te', 'DeviceType_te', 'Transaction_day_of_week_te', 'P_emaildomain_2_te',
               'firstInRow_te', 'TransactionAmt_mod1']

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
# ----------------------------------------- CATBOOST -----------------------------------------------

# ----- Categorical Features for CB
categorical_features = categorical_features + ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
categorical_features = categorical_features + ['M'+str(i) for i in range(1, 10)]
categorical_features = categorical_features + ['id_'+str(i) for i in range(12, 39)]
categorical_features = categorical_features + ['DeviceType', 'DeviceInfo', 'id_04']
categorical_features = categorical_features + ['device_name', 'device_version', 'OS_id_30', 'version_id_30',
                                                'browser_id_31', 'version_id_31', 'screen_width', 'screen_height',
                                                'AmtCentsLen', 'Transaction_day_of_week', 'Transaction_hour', 'isFirst']
categorical_features = categorical_features + ['UserID_base', 'UserID_mid', 'UserID_full']
categorical_features = categorical_features + [f for f in X_dev.columns if '_oh_' in f]


categorical_features = list(set(categorical_features).intersection(set(X_dev.columns)))


params = {'loss_function': 'Logloss',
          'eval_metric': 'AUC',
          'iterations': cb_iters,
          'depth': 8,
          'use_best_model': True,
          'cat_features': categorical_features,
          # 'ignored_features': cols_to_drop,
          'early_stopping_rounds': early_stopping_rounds,
          'verbose': 100,
          'random_seed': 322,
          'eta': 0.01,
          'scale_pos_weight': 28,
          # 'one_hot_max_size': 1024,
          'rsm': 0.8,
          'thread_count': 32
          }

cb = CatBoostClassifier(**params)

cb.fit(X_dev, y_dev, eval_set=(X_val, y_val), use_best_model=True)
print("-"*30 + 'VALIDATION' + "-"*30)
print(cb.best_score_['validation'])
print("-"*70)
val_score = np.round(cb.best_score_['validation']['AUC'], 5)

if DoForecast:
    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- FULL CB ------------------------------------------------
    params['iterations'] = cb.best_iteration_
    params['use_best_model'] = False
    params['early_stopping_rounds'] = None

    y_dev = train['isFraud']
    X_dev = train.drop(columns='isFraud')
    del X_dev['TransactionDT']
    del X_dev['TransactionID']

    cb = CatBoostClassifier(**params)
    cb.fit(X_dev, y_dev, use_best_model=False, plot=False)

    # --------------------------------------------------------------------------------------------------
    # ----------------------------------------- PREDICTION ---------------------------------------------

    del test['TransactionDT']
    ids = test['TransactionID']
    del test['TransactionID']

    print(X_dev.shape, test.shape)

    preds = cb.predict_proba(test)

    y_preds = pd.DataFrame(columns=['TransactionID'])
    y_preds['TransactionID'] = ids
    y_preds['isFraud'] = [p[1] for p in preds]

    print("Saving submission file")
    script_name = os.path.basename(__file__).split('.')[0]
    y_preds[['TransactionID', 'isFraud']].to_csv('{}__{}.csv'.format(script_name, str(val_score)), index=False)



