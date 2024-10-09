import pandas as pd
import numpy as np
import os
import sys

if not os.path.exists('results'):
    os.makedirs('results')

def similarity(ds):
    count_df = ds.value_counts()
    num = 0 if 0 not in count_df.index else count_df[0]
    num += 0 if 1 not in count_df.index else count_df[1]/4.0
    num += 0 if -1 not in count_df.index else count_df[-1]/4.0
    num += 0 if 1000 not in count_df.index else count_df[1000]/4.0
    num += 0 if -1000 not in count_df.index else count_df[-1000]/4.0
    num /= len(ds)
    if num == 0:
        num += 1e-4
    return num

def match(ds):
    num = ds.value_counts()[0]
    return num/len(ds)

def model(cityname, isVal, start_day, end_day, fill_version, conf_denum, conf_xpow):

    output_dir = 'results/val' if isVal else 'results/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_filename = 'split_data_%d_%d/v%d/city%s_masked_train_data.csv.gz'% (start_day, end_day-start_day,fill_version,cityname) if isVal\
        else 'filled_data/v%d/city%s_masked_train_data.csv.gz'%(fill_version,cityname)

    start_step = start_day*48
    end_step = end_day*48

    task_data = pd.read_csv(data_filename, compression='gzip')

    conf_filename = 'split_conf_%d_%d/city%s_masked_train_confidence_%d_%.1f.csv.gz'% (start_day, end_day-start_day,cityname,conf_denum,conf_xpow) if isVal\
        else 'confidence_data/city%s_confidence_%d_%.1f.csv.gz'% (cityname,conf_denum,conf_xpow)

    task_conf = pd.read_csv(conf_filename, compression='gzip')

    val_agent = task_data.loc[task_data['%d'%(start_step)] == 999999].index

    for i in range(start_step,end_step):
        selected_cols = ['%d'%(i-7*k*48) for k in range(1,12) if 0 <= i-7*k*48 < start_step]
        p_conf = task_conf[selected_cols].copy()
        p_loc = task_data[selected_cols]
        loc_sim = p_loc.copy().sub(task_data['%d'%i], axis = 0)
        loc_sim = loc_sim.apply(similarity)
        multiplied_conf = p_conf.copy().mul(loc_sim, axis = 1)

        # p_conf = task_conf[selected_cols]
        # p_loc = task_data[selected_cols]
        # loc_sim = p_loc.sub(task_data['%d'%i], axis = 0).apply(similarity)
        # multiplied_conf = p_conf.mul(loc_sim, axis = 1)

        for aidx in val_agent:
            find_df = pd.DataFrame()
            find_df['loc'] = p_loc.iloc[aidx]
            find_df['conf'] = multiplied_conf.iloc[aidx]
            column_idx = find_df.groupby('loc').cumsum().idxmax().item()
            task_data.loc[aidx, '%d'%i] = p_loc.loc[aidx, column_idx]

            if aidx%1000 == 0: print('step %d: %dk agent done'%(i, aidx/1000))




    task_data.iloc[val_agent, start_step:end_step].to_csv('%s/city%s_s%d_v%d_%d_%.1f.csv.gz'%(output_dir,cityname,start_day,fill_version,conf_denum,conf_xpow), compression='gzip', index=False)

if __name__ == '__main__':

    city = 'D'
    fill_version = -1
    conf_denum = 0
    conf_xpow = 0
    start_day = 0
    isVal = True

    for i in range(1,len(sys.argv)):
        if sys.argv[i] == '-val':
            isVal = True
            continue

        if sys.argv[i] == '-city':
            city = sys.argv[i+1]
        if sys.argv[i] == '-v':
            fill_version = int(sys.argv[i+1])
        if sys.argv[i] == '-denum':
            conf_denum = int(sys.argv[i+1])
        if sys.argv[i] == '-pow':
            conf_xpow = float(sys.argv[i+1])
        if sys.argv[i] == '-start.day':
            start_day = int(sys.argv[i+1])
        i += 1

    end_day = 60 if isVal else 75
    end_day = 75 if city == 'A' else end_day
    if not isVal: start_day = 60

    model(city, isVal, start_day, end_day, fill_version, conf_denum, conf_xpow)