import numpy as np
import pandas as pd
import geobleu
import os
import sys

def reformat_val(cityname,start_day):
    raw_data = pd.read_csv('raw_data/city%s_challengedata.csv.gz'%cityname, compression='gzip')
    val_agent = raw_data[raw_data['x'] == 999]['uid'].unique()
    val_data = raw_data[raw_data['uid'].isin(val_agent)]
    val_data = val_data[val_data['d'] >= start_day]
    val_data = val_data[val_data['d'] < 60].reset_index(drop=True)
    return val_data

def reformat_result(cityname, start_day, fill_version, conf_denum, conf_xpow):


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

    return merged


def cal_acc(merged):
    merged['ori'] = merged['x_x']*1000 + merged['y_x']
    merged['gen'] = merged['x_y']*1000 + merged['y_y']
    merged['check'] = merged['gen'] - merged['ori']
    return merged['check'].value_counts()[0]/ len(merged)

def cal_bleu(val_data, merged):
    generated = merged[['uid','d','t','x_y','y_y']]
    generated = generated.rename(columns={'x_y':'x','y_y':'y'})

    generated_group = generated.groupby('uid')
    val_group = val_data.groupby('uid')

    geobleu_val = 0
    unique_id = generated['uid'].unique()
    for aid in unique_id:
        geobleu_val += geobleu.calc_geobleu(generated_group.get_group(aid).to_numpy(), val_group.get_group(aid).to_numpy(), processes=3)
    return geobleu_val/len(unique_id)

if __name__ == '__main__':
    city = 'D'
    job = 'acc'
    denums = [24] * 8 + [18,28]
    xpows = [10,8,5,3,1.5,1,0.7,0.5,0.7,0.7]
    starts = [45,50]

    # print('City,Validation Start Day,Filling Strategy,Denominator,Power,Accuracy')

    for start_day in starts:
        val_data = reformat_val(city,start_day)

        for fill_version in range(8,9):
            for conf_denum, conf_xpow in zip(denums,xpows):
                reformat_data = reformat_result(city, start_day, fill_version, conf_denum, conf_xpow)
                acc = cal_acc(reformat_data)
                print('%s,%d,%d,%d,%.1f,%.8f'%(city, start_day, fill_version, conf_denum, conf_xpow,acc))
