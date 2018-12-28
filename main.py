from Processer import *
from config imprort *
from dae_model import *
from generator import *

def load_data():
    
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    
    df = pd.concat([df_train, df_test], axis=0)
    df = df.fillna(-1)
    
    cat_list = []
    calc_list = []
    no_calc_list = []
    non_features_list = ['id','target']
    
    for col in df.columns:
        if col in non_features_list:
            continue
        if re.search(r'calc', col):
            calc_list.append(col)
        else:
            no_calc_list.append(col)
            if re.search(r'cat$', col):
                cat_list.append(col)
    
    transformer_one = [
    (Processer.drop_columns, dict(col_names=calc_list)),
    (Processer.drop_columns, dict(col_names=non_features_list)),
    (Processer.dtype_transform, dict()),
    (Processer.onehot,dict(cat_features = cat_list,drop_origin=False)),
    (Processer.rank_guass,dict(col_names = no_calc_list)),
    ]
    df = Compose(transformer_one)(df)
    
    print("the shape of dataframe : {0}".format(df.shape))
    
    return df

 def main():
     
     df = load_data()
     nn = NN_model(df)
     nn.fit()
main()
