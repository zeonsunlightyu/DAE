def rank_guass_v_one(df,col_name):
       
    series = df[col_name].rank()
    series = series.values
    M = series.max()
    m = series.min() 
    series = ((series-m)/ (M - m))*1.98 - 0.99
    series = np.sqrt(2)*erfinv(series)
    series = series - series.mean()    
    return pd.Series(series)
    
def rank_gauss(x):
    
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    
    return efi_x
    
def rank_guass_rm_dupliacted(x):
    x_temp = pd.Series(x).drop_duplicates().values
    N = x_temp.shape[0]
    temp = x_temp.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    value_dict = dict(zip(x_temp, efi_x))
    return pd.Series(x).map(value_dict)
