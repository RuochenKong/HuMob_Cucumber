import pandas as pd
import numpy as np
import os
import sys


if not os.path.exists('split_conf_60_15'):
    os.makedirs('split_conf_60_15')

def split_data(cityname,denum, xpow):

    val_cols = ['%d'%i for i in range(60*48,75*48)]
    conf_df = pd.read_csv('confidence_data/city%s_confidence_%d_%.1f.csv.gz'%(cityname, denum, xpow), compression='gzip')
    conf_df.loc[conf_df.index[-3000:], val_cols].to_csv('split_conf_60_15/city%s_validation_confidence_%d_%.1f.csv.gz'%(cityname, denum, xpow), compression='gzip', index=False)
    conf_df.loc[conf_df.index[-3000:], val_cols] = -1
    conf_df.to_csv('split_conf_60_15/city%s_masked_train_confidence_%d_%.1f.csv.gz'%(cityname,denum, xpow), compression='gzip', index=False)

if __name__ == '__main__':
    city = sys.argv[1]
    denum = int(sys.argv[2])
    xpow = float(sys.argv[3])
    split_data(city,denum, xpow)