import pandas as pd
import numpy as np
import os
import sys


if not os.path.exists('split_data_60_15'):
    os.makedirs('split_data_60_15')

def split_data(cityname,version):

    outdir = 'split_data_60_15/v%d'%version
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_df = pd.read_csv('filled_data/v%d/city%s_data.csv.gz'%(version,cityname), compression='gzip')


    # data_df[['%d'%i for i in range(60*48,75*48)]].to_csv('%s/city%s_test_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)

    val_cols = ['%d'%i for i in range(60*48,75*48)]
    data_df.loc[data_df.index[-3000:], val_cols].to_csv('%s/city%s_validation_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)

    data_df.loc[data_df.index[-3000:], val_cols] = 999999
    data_df.to_csv('%s/city%s_masked_train_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)

if __name__ == '__main__':
    city = sys.argv[1]
    version = int(sys.argv[2])

    split_data(city,version)