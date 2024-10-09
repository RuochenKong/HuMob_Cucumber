import pandas as pd
import numpy as np
import os
import sys

if not os.path.exists('processed_data'):
    os.mkdir('processed_data')

def reform_time_and_loc(cityname):
    data_type = 'groundtruth' if cityname == 'A' else 'challenge'
    df = pd.read_csv('raw_data/city%s_%sdata.csv.gz'%(cityname,data_type), compression='gzip')
    df['step'] = df['d']*48 + df['t']
    df['loc'] = df['x']*1000 + df['y']
    df[['uid', 'step', 'loc']].to_csv('processed_data/city%s_data.csv.gz'%cityname, compression='gzip', index=False)

if __name__ == '__main__':
    city = sys.argv[1]
    reform_time_and_loc(city)
