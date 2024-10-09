import numpy as np
import pandas as pd
import geobleu
import os
import sys


def reformat(cityname, start_day, fill_version, conf_denum, conf_xpow):
    fn  = 'raw_data/city%s_challengedata.csv.gz'%cityname if cityname != 'A' else 'raw_data/cityA_groundtruthdata.csv.gz'
    raw_data = pd.read_csv(fn, compression='gzip')
    val_agent = raw_data['uid'].unique()[-3000:]
    val_data = raw_data[raw_data['uid'].isin(val_agent)]
    val_data = val_data[val_data['d'] >= 60].reset_index(drop=True)

    raw_result = pd.read_csv('results/val/city%s_s%d_v%d_%d_%.1f.csv.gz'%(cityname,start_day,fill_version,conf_denum,conf_xpow), compression='gzip').to_numpy()
    reformat = {'uid':[], 'd':[], 't':[], 'x':[], 'y':[]}

    start_test_id = val_data['uid'].min()
    for i in range(len(raw_result)):
        loc = -1
        for j in range(len(raw_result[0])):
            d = int(j/48) + start_day
            t = j%48
            loc = raw_result[i][j]
            x = int(loc/1000)
            y = loc%1000
            reformat['x'].append(x)
            reformat['y'].append(y)
            reformat['t'].append(t)
            reformat['d'].append(d)
            reformat['uid'].append(i+start_test_id)
    reformat = pd.DataFrame(reformat)
    merged = pd.merge(val_data, reformat, on=['uid','d','t'], how='left')

    return val_data, merged


def cal_acc(merged):
    merged['ori'] = merged['x_x']*1000 + merged['y_x']
    merged['gen'] = merged['x_y']*1000 + merged['y_y']
    merged['check'] = merged['gen'] - merged['ori']
    print('Accuracy: %.8f'%(merged['check'].value_counts()[0]/ len(merged)))
    return merged['check'].value_counts()[0]/ len(merged)

def cal_bleu(val_data, merged, ifDTW = False):
    generated = merged[['uid','d','t','x_y','y_y']]
    generated = generated.rename(columns={'x_y':'x','y_y':'y'})

    generated_group = generated.groupby('uid')
    val_group = val_data.groupby('uid')

    geobleu_val = 0
    dtw_val = 0
    unique_id = generated['uid'].unique()
    for aid in unique_id:
        geobleu_val += geobleu.calc_geobleu(generated_group.get_group(aid).to_numpy(), val_group.get_group(aid).to_numpy(), processes=3)
        if ifDTW: dtw_val += geobleu.calc_dtw(generated_group.get_group(aid).to_numpy(), val_group.get_group(aid).to_numpy(), processes=3)
        # if aid%100 == 0: print('\t partial sum Bleu:', aid, geobleu_val)
    print('Bleu: %.8f'%(geobleu_val/len(unique_id)))
    if ifDTW: print('DTW: %.8f'%(dtw_val/len(unique_id)))

    return geobleu_val/len(unique_id)


if __name__ == '__main__':
    city = 'A'
    fill_version = 2
    conf_denum = 28
    conf_xpow = 0.7
    start_day = 60

    val_data, reformat_data = reformat(city, start_day, fill_version, conf_denum, conf_xpow)
    cal_acc(reformat_data)
    cal_bleu(val_data, reformat_data, True)