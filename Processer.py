import numpy as np
import pandas as pd
from scipy.special import erfinv
import re
    
def drop_columns(df, col_names):
        
    print('Before drop columns {0}'.format(df.shape))
    df = df.drop(col_names, axis = 1)
    print('After drop columns {0}'.format(df.shape))
    return df
   
def dtype_transform(df):
        
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int8)
    return df

def label_encode(df,col_names):
       
    df = preprocessing.LabelEncoder().fit(df[col_names]).transform(df[col_names])
        
    return df

def fill_missing(df,col_names,fill_values):
       
    for i in range(len(col_names)):
        print('filling {0}'.format(col_names[i]))
        df[col_names[i]] = fill_values[i].fit(df[col_names[i]]).transform(df[col_names[i]])
        
    return df
 
def onehot(df,cat_features,drop_origin = True,threshold=0):
        
    print('Before ohe : dataframe {0}'.format(df))
    for column in cat_features:
        print("the number of unique number(exclude threhold): {0}".format(len(df[column].unique())))
        dummy_column = pd.get_dummies(pd.Series(df[column]), prefix=column)
        abort_columns = []
        for col in dummy_column:
            if dummy_column[col].sum() < threshold:
                print('column {0} unique value {1} less than threshold {2}'.format(col,dummy_column.sum(),threshold))
                abort_columns.append(col)
        print("Abort columns : {0}".format(abort_columns))
        remain_cols = [c for c in dummy_column.columns if c not in abort_columns]
        df = pd.concat([df,dummy_column[remain_cols]],axis=1)
        if drop_origin:
            print("Drop column : {0}".format(column))
            df = df.drop([column], axis=1)
    return df
    
def rank_guass(df,col_names):
       
    for c in col_names:
        series = df[c].rank()
        M = series.max()
        m = series.min() 
        print(c, m, len(series), len(set(df[c].tolist())))
        series = (series-m)/(M-m)
        series = series - series.mean()
        series = series.apply(erfinv) 
        df[c] = series
            
    return df

class Compose(object):
    def __init__(self, transforms_params):
        self.transforms_params = transforms_params
    def __call__(self, df):
        for transform_param in self.transforms_params:
            transform, param = transform_param[0], transform_param[1]
            df = transform(df, **param)
        return df
