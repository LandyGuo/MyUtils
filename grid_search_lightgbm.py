#coding=utf-8
import sys
import copy
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score

grid_search_params = {"num_leaves":[60],
               "learning_rate":[0.1],
               "feature_fraction":[0.95],
               "min_data_in_leaf":[10],
               "max_bin":[15],
               "boosting_type":['gbdt']
		}

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': None,
    'learning_rate': None,
    'feature_fraction': None,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'early_stopping_rounds':5,
    'is_unbalance':True,
    'num_threads':40,
}

lr = 0.25
num_l = 55
ff = 0.9


save_root = "root"
root = "workspace"

def load_train_test(limit_train=100000000000):
    print('Load data...')

    df_train = pd.read_csv(root + '/train.txt', header=None, sep='\t')
    df_test = pd.read_csv(root + '/test.txt', header=None, sep='\t')

    y_train = df_train[0].values[:limit_train]
    y_test = df_test[0].values
    X_train = df_train.drop([0], axis=1).values[:limit_train]
    X_test = df_test.drop([0], axis=1).values

    num_train, num_feature1 = X_train.shape
    print("Num of Train sample: {}".format(num_train))
    print("Num of Feat: {}".format(num_feature1))

    num_test, num_feature2 = X_test.shape
    print("Num of Test sample: {}".format(num_test))
    print("Num of Feat: {}".format(num_feature2))

    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
    feature_name = ['feature_' + str(col) for col in range(num_feature1)]
    return lgb_train, lgb_eval,  feature_name


def parse_params(val_dic, get_params):
    return "_".join([str(val_dic[x]) for x in get_params])

def recursive_set_param(ret, params, grid_search_params, get_params, params_set):
#         print "grid_search_params:", grid_search_params
    if not grid_search_params:
        x = parse_params(params, get_params)
        if x not in params_set:
            ret.append(copy.deepcopy(params))
        params_set.add(x)
        return 
    for k in grid_search_params:
        if k in params:
            origin_values = grid_search_params[k]
            for origin_value in origin_values:
                params[k] = origin_value
                if k in grid_search_params:
                    del grid_search_params[k]
                recursive_set_param(ret, params, grid_search_params, get_params, params_set)
            grid_search_params[k] = origin_values

def get_training_params_with_grid_search(params, grid_search_params):
    get_params = grid_search_params.keys()                     
    ret, params_set = [], set()
    recursive_set_param(ret, params, grid_search_params, get_params, params_set)
    return ret



# load data
lgb_train, lgb_eval, feature_name = load_train_test(limit_train=10000)


training_params = get_training_params_with_grid_search(params, grid_search_params)
print len(training_params)

for training_p in training_params: # begin training
    p_names = grid_search_params.keys()
    model_name = parse_params(training_p, p_names)
    x = model_name.split('_')
    print "params:", [(a,b) for a,b in zip(p_names, x)]


    print('Start training...')
    gbm = lgb.train(training_p,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=lgb_eval,  # eval training data
                    feature_name=feature_name)
    
    save_model_path = save_root+"/"+model_name+'.txt'
    # save model to file
    gbm.save_model(save_model_path)


    # load model to predict
    bst = lgb.Booster(model_file=save_model_path)
    # can only predict with the best iteration (or the saving iteration)
    X_test, y_test = lgb_eval.data, lgb_eval.get_label()
    y_pred = bst.predict(X_test)
    # eval with loaded model
    print "result precision:", precision_score(y_test, y_pred>0.5, average=None)
    print "result recall:", recall_score(y_test, y_pred>0.5, average=None)
    print "--------------------------------------------------"
    sys.stdout.flush()     






