import pandas as pd
import numpy as np
import os
import sys

if not os.path.exists('filled_data'):
    os.mkdir('filled_data')

outdir = 'filled_data/check'
if not os.path.exists(outdir):
    os.mkdir(outdir)

def most_common_except(row):
    if row.name % 1000 == 0:  print ('%dk agent finished step 2'%int(row.name/1000))
    filtered_row = row[(row != -1) & (row != 999999)]
    if not filtered_row.empty:
        return filtered_row.mode().iloc[0]
    return None

def fill_with_most_common(row):
    if row.name % 1000 == 0:  print ('%dk agent finished step 3'%int(row.name/1000))
    most_common = row['most_common']
    return row.replace(-1, most_common)

def filling_and_cal_conf(cityname):
    df = pd.read_csv('processed_data/city%s_data.csv.gz'%cityname, compression='gzip')
    max_uid = df['uid'].max() + 1
    full_loc = np.ones((max_uid, 75*48), dtype=int) * -1

    for uid,udf in df.groupby('uid'):
        steps = list(udf['step'])
        locs = list(udf['loc'])
        last_step = len(steps) - 1
        is_test = False

        if uid % 1000 == 0:  print ('%dk agent finished step 1'%int(uid/1000))

        for i in range(last_step):
            if locs[i] == 999999:
                full_loc[uid, 60*48:] = 999999
                is_test = True
                break

            nextStep = min(steps[i+1], steps[i] + 24)
            full_loc[uid, steps[i]: nextStep] = locs[i]

            # if nextStep - steps[i] == 1:
            #     continue

        # last one
        if is_test: continue
        nextStep =  min(steps[last_step] + 24, 3600)
        full_loc[uid, steps[last_step]: nextStep] = locs[last_step]
        # full_loc[steps[last_step]: , uid] = locs[last_step]

    print('-'*50)
    full_df = pd.DataFrame(full_loc)
    full_df.columns = ['%d'%i for i in range(0,3600)]

    full_df['most_common'] = full_df.apply(most_common_except, axis=1)
    print('-'*50)

    full_df.iloc[:, :-1] = full_df.apply(fill_with_most_common, axis=1)
    full_df = full_df.drop(columns=['most_common'])

    full_df.to_csv('%s/city%s_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)


    return full_df

if __name__ == '__main__':
    city = sys.argv[1]
    filling_and_cal_conf(city)