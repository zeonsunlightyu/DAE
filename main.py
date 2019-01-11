from Processer import *
from config import *
from dae_model import *
from generator import *

def load_data():
    
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    
    df = pd.concat([df_train, df_test], axis=0)
    
    cat_list = []
    calc_list = []
    rankgauss_list = []
    non_features_list = ['id','target']
    binary_features = []
    for col in df.columns:
        if col in non_features_list:
            continue
        if re.search(r'calc', col):
            calc_list.append(col)
        else:
            if re.search(r'cat$', col):
                cat_list.append(col)

    for col in df.columns:
        if col in non_features_list or col in calc_list:
            continue
        if len(df[col].value_counts()) > 2 :
            rankgauss_list.append(col)
        else:
            binary_features.append(col)

    print("the cat list : {0}".format(cat_list))
    print("the calc list : {0}".format(calc_list))
    print("the rank_guass list : {0}".format(rankgauss_list))
    print("the binary feature list : {0}".format(binary_features))
    transformer = [
    (drop_columns, dict(col_names=calc_list)),
    (drop_columns, dict(col_names=non_features_list)),
    (onehot,dict(cat_features = cat_list,drop_origin=False))
    ]
   
    df = Compose(transformer)(df)
    df_train = df[:len(df_train)]
    df_test = df[len(df_train):]
    
    for col in rankgauss_list:
        print(col)
        rankgauss_map = build_rankgauss_trafo(df_train[col])
        df_train[col] = apply_rankgauss_map(df_train[col],rankgauss_map)
        df_test[col] = apply_rankgauss_map(df_test[col],rankgauss_map)
        
    df = pd.concat([df_train, df_test], axis=0)
    
    print("the shape of dataframe : {0}".format(df.shape))
    show_dataframe_stats(df)
    
    return df

def main():
    np.random.seed(12345)
    df = load_data()
    nn = NN_model(df.values)
    nn.fit_generator()
main()
