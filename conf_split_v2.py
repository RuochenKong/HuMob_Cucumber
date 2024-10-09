import pandas as pd
import numpy as np
import os
import sys


if not os.path.exists('split_conf_50_10'):
    os.makedirs('split_conf_50_10')

def split_data(cityname,denum, xpow):

    data_df = pd.read_csv('filled_data/v1/city%s_data.csv.gz'%(cityname), compression='gzip')
    val_agent = data_df.loc[data_df['%d'%(60*48)] == 999999].index
    val_cols = ['%d'%i for i in range(50*48,60*48)]

    conf_df = pd.read_csv('confidence_data/city%s_confidence_%d_%.1f.csv.gz'%(cityname, denum, xpow), compression='gzip')
    conf_df.loc[val_agent, val_cols].to_csv('split_conf_50_10/city%s_validation_confidence_%d_%.1f.csv.gz'%(cityname, denum, xpow), compression='gzip', index=False)
    masked_train = conf_df[['%d'%i for i in range(60*48)]].reset_index(drop=True)
    masked_train.loc[val_agent, val_cols] = -1
    masked_train.to_csv('split_conf_50_10/city%s_masked_train_confidence_%d_%.1f.csv.gz'%(cityname,denum, xpow), compression='gzip', index=False)

if __name__ == '__main__':
    city = sys.argv[1]
    denum = int(sys.argv[2])
    xpow = float(sys.argv[3])
    split_data(city,denum, xpow)