import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer
from scipy import stats

def discrization_by_xgb(df, fea, bst):
    df_fea = df[fea]
    # transform according the split value
    split_interval = bst.get_split_value_histogram(fea, as_pandas=True)['SplitValue']
    if split_interval[0] > 0:
        split_interval = pd.concat([pd.Series([0]), split_interval], ignore_index=True)
    split_interval = pd.concat([pd.Series([float("-inf")]), split_interval], ignore_index=True)
    split_interval = pd.concat([split_interval, pd.Series([float("inf")])], ignore_index=True)
    print(split_interval)
    result = pd.cut(df_fea, split_interval, labels=range(split_interval.shape[0] - 1))

    return result

def con_to_cat_fea(df, fea, method, bins):

    fea = np.array(df[fea]).reshape(-1, 1)
    k_binner = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=method).fit(fea)
    fea = k_binner.transform(fea)
    fea = pd.Series(fea.reshape(-1))

    return fea, k_binner.n_bins_

def cdf_transform_discrization(fea,n_bins):
    return fea/(n_bins-1)

def eliminate_low_frequncy_cate(df,fea,min_obs):

    df_fea = df[fea]
    # replace low frequnce cat value with nan
    val = dict((k, np.nan if v < min_obs else k) for k, v in dict(Counter(df_fea)).items())
    k, v = np.array(list(zip(*sorted(val.items()))))
    df_fea = v[np.digitize(df_fea, k, right=True)]

    return df_fea

def onehot_encode(df,fea,min_obs,dummy_na,sparse):

    df_fea = eliminate_low_frequncy_cate(df, fea, min_obs)

    return pd.get_dummies(df_fea, dummy_na=dummy_na, sparse=sparse,prefix = fea)

def label_encode(df,fea,min_obs):

    df_fea = preprocessing.LabelEncoder().fit(df[fea]).transform(df[fea])
    #replace low frequnce cat value with nan
    df_fea = eliminate_low_frequncy_cate(df, fea, min_obs)

    return df_fea

#def remove_outlier(df,three_sigma = True):

#def three_sigma_remove(df):
#def iqr_remove(df,fea):

def missing_value_handling(df,fea,method,fill_value = None):
    df_fea = df[fea]
    imp = SimpleImputer(missing_values=np.nan, strategy=method,fill_value=fill_value)
    imp.fit(df_fea)
    return imp.transform(df_fea)

def normalization(df,fea,method,n_bins = 10):
    df_fea = df[fea]
    if method == 'standard':
        df_fea = preprocessing.StandardScaler().fit_transform(df_fea)

    if method == 'minmax':
        df_fea = preprocessing.MinMaxScaler().fit_transform(df_fea)

    if method == 'cdf':
        df_fea = con_to_cat_fea(df, fea, method, n_bins)
        df_fea = cdf_transform_discrization(df_fea, n_bins)

    return df_fea

def transformation(df,fea):
    transformed = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(
        df[fea].reshape(-1, 1))
    return pd.Series(transformed.reshape(-1))

def cross_validation(train, test, params, label_name, feature_list, display, do_error_analysis,num_boost_round):
    if display:
        print('show the params----')
        print(params)

    NFOLDS = 5

    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

    y_train = train[label_name]
    X_train = scipy.sparse.csr_matrix(train[feature_list].values)

    cv_train = np.zeros(len(y_train))

    final_cv_train = np.zeros(len(y_train))
    final_best_trees = []

    kf = kfold.split(X_train, y_train)

    best_trees = []
    fold_scores = []
    models = []

    if do_error_analysis:
        error_analysis_dfs = []

    for i, (train_fold, validate) in enumerate(kf):

        if test != None:
            cv_pred = np.zeros(len(test))
            final_cv_pred = np.zeros(len(test))

        X_train_fold, X_validate_fold, y_train_fold, y_validate_fold = \
            X_train[train_fold, :], X_train[validate, :], y_train[train_fold], y_train[validate]

        bst = train_xgboost(X_train_fold, y_train_fold, X_validate_fold, y_validate_fold, params, feature_list,
                            num_boost_round,False)
        if display:
            xgb.plot_importance(bst, color='red')
            plt.show()

        best_trees.append(bst.best_iteration)

        if test != None:
            cv_pred += bst.predict(xgb.DMatrix(test))

        cv_train[validate] += bst.predict(xgb.DMatrix(X_validate_fold, feature_names=feature_list),
                                          ntree_limit=bst.best_ntree_limit)
        score = roc_auc_score(y_validate_fold, cv_train[validate])

        if display:
            print(score)

        models.append(bst)
        fold_scores.append(score)

    if display:
        print("cv score:")
        print(roc_auc_score(y_train, cv_train))
        print(fold_scores)
        print(np.mean(fold_scores))
        print(best_trees, np.mean(best_trees))

    return models


def train_xgboost(X_train, Y_train, X_valid, Y_valid, params, feature_list,num_boost_round,generate_valid,val_size):

    if generate_valid :
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = val_size, random_state = 42)
        dtrain = xgb.DMatrix(X_train, Y_train, feature_names=feature_list)
        dvalid = xgb.DMatrix(X_val, Y_val, feature_names=feature_list)

    else:
        dtrain = xgb.DMatrix(X_train, Y_train, feature_names=feature_list)
        dvalid = xgb.DMatrix(X_valid, Y_valid, feature_names=feature_list)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, verbose_eval=20, \
                    early_stopping_rounds=25, maximize=True)
    return bst

def train_test_split_train(X_train, Y_train, X_valid, Y_valid, params, feature_list,num_boost_round,split):

    bst = train_xgboost(X_train, Y_train, X_valid, Y_valid, params, feature_list,num_boost_round,split)

    predicted = bst.predict(xgb.DMatrix(X_valid, feature_names=feature_list),
                ntree_limit=bst.best_ntree_limit)

    score = roc_auc_score(Y_valid,predicted)

    return score

def forward_elimination(df,features,test,params,min_imp):
    feature_list = list(df[features].columns)
    
    df_ = df[features]
    y = df['y']
    
    feature_select_count = {}
    rand_seed_list = [42,2018]
    
    for rand in rand_seed_list:
        
        selected_list = []
        best_score = 0
        random.seed(rand)
        random.shuffle(feature_list)
        
        for feature in feature_list :
            
            print("----------------------------------------------------------------------------------------------")
            temp_list = list(selected_list)
            temp_list.append(feature)
            score = train_test_split_train(df_[temp_list],y,test[temp_list],test['y'],params,temp_list,50,False,0.5)
            
            print("currently checking....",feature)
            if (score-best_score) > min_imp:
                print ("current_score :",score)
                print ("current_features :",temp_list)
                best_score = score
                print ("best_score :",best_score)
                selected_list.append(feature)
        for key in selected_list:
            if feature_select_count.get(key) :
                feature_select_count[key] = feature_select_count[key] + 1
            else:
                feature_select_count[key] = 1
        
        print("----------------------------------------------------------------------------------------------")
                    
    return selected_list,feature_improve,feature_select_count


def backward_elimination(df,features,test,params,max_cut):
    
    feature_list = list(df[features].columns)
    
    df_temp = df[features]
    y = df['y']
    
    feature_select_count = {}
    rand_seed_list = [42,2018]
    
    for rand in rand_seed_list:
        
        selected_list = list(feature_list)
        best_score = train_test_split_train(df_temp[selected_list],y,test[selected_list],
                                            test['y'],params,selected_list,50,False,0.5)
        random.seed(rand)
        random.shuffle(feature_list)
        
        for feature in feature_list :
            
            print("----------------------------------------------------------------------------------------------")
            temp_list = list(selected_list)
            temp_list.remove(feature)
            score = train_test_split_train(df_temp[temp_list],y,test[temp_list],
                                           test['y'],params,temp_list,50,False,0.5)
            
            print("currently checking....",feature)
            if (best_score - score) < max_cut:
                print ("current_score :",score)
                best_score = score
                print ("best_score :",best_score)
                selected_list.remove(feature)
                
        for key in selected_list:
            if feature_select_count.get(key) :
                feature_select_count[key] = feature_select_count[key] + 1
            else:
                feature_select_count[key] = 1
        
    return selected_list,feature_improve,feature_select_count
    

def feature_one_by_one(df,features,test,params):
    
    feature_list = list(df[features].columns)
    
    df_temp = df[features]
    y = df['y']
    
    feature_score = {}
    
    for feature in feature_list :
        print("----------------------------------------------------------------------------------------------")
        score = train_test_split_train(df_temp[[feature]],y,test[[feature]],
                                           test['y'],params,[feature],50,False,0.5)
         
        feature_score[feature] = score
        print("current feature  :", feature)
        print("score :", score)
        
    return sorted(feature_score.items(), key=operator.itemgetter(1))
        
def add_swap_noise(cur_data, all_data, noise_level=0.07):
    """
    Add swap noise to current data
    :param cur_data: Current batch of data
    :param all_data: The whole data set
    :param noise_level: percentage of columns being swapped
    :return: data with swap noise added
    """
    batch_size = cur_data.shape[0]
    num_samples = all_data.shape[0]
    num_features = cur_data.shape[1]
    #可能会导致同一个batch内部的互换,虽然概率应该很低
    random_row = np.random.randint(0, num_samples, size=batch_size)
    for i in range(batch_size):
        random_swap = np.random.rand(num_features) < noise_level
        cur_data[i, random_swap] = all_data[random_row[i], random_swap]
    return cur_data
