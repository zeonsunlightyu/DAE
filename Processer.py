class Processer(object):
    """
    dataframe 处理类的方法
    """
    def drop_columns(df, col_names):
        """
        删除不需要的列
        df : 类型为Dataframe的数据集
        col_names : 需要删除的列的列表
        return : 删掉不需要列之后的Dataframe
        """
        
        print('Before drop columns {0}'.format(df.shape))
        df = df.drop(col_names, axis = 1)
        print('After drop columns {0}'.format(df.shape))
        return df
    
    def dtype_transform(df):
        """
        数据类型转换
        df : 类型为Dataframe的数据集
        return：数据类型转换之后的数据集
        """
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype(np.int8)
        return df
    
    def label_encode(df,col_names):
        """
        对非数值的特征进行编码
        df: 类型为Dataframe的数据集
        col_names : 需要转换的特征列表
        """
        df = preprocessing.LabelEncoder().fit(df[col_names]).transform(df[col_names])
        
        return df
    
    def fill_missing(df,col_names,fill_values):
        #SimpleImputer(missing_values=np.nan, strategy=method,fill_value=fill_value)
        """
        缺失值填充
        df: 类型为Dataframe的数据集
        col_names : 需要被填充的特征列表
        fill_values : 每个特征的填充值或者方法
        """
        for i in range(len(col_names)):
            print('filling {0}'.format(col_names[i]))
            df[col_names[i]] = fill_values[i].fit(df[col_names[i]]).transform(df[col_names[i]])
        
        return df
                
    """
    变量变换类的方法（离散化，onehot，）
    """
    
    def onehot(df,cat_features,drop_origin = True,threshold=5):
        """
        one hot encoding
        df:类型为Dataframe的数据集
        cat_features：需要转换的离散变量名称
        drop_origin：
        return: 离散变量都变成onehot的数据集
        """
        print('Before ohe : dataframe {0}'.format(df))
        for column in cat_features:
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
                df = df.drop([column], axis=1)
        return df
