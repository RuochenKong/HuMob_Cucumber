import pandas as pd
import numpy as np
import os
import sys

if not os.path.exists('filled_data'):
    os.mkdir('filled_data')

outdir = 'filled_data/v2'
if not os.path.exists(outdir):
    os.mkdir(outdir)


def filling_and_cal_conf(cityname):
    df = pd.read_csv('processed_data/city%s_data.csv.gz'%cityname, compression='gzip')
    max_uid = df['uid'].max() + 1
    full_loc = np.ones((75*48,max_uid), dtype=int) * -1

    for uid,udf in df.groupby('uid'):
        steps = list(udf['step'])
        locs = list(udf['loc'])
        last_step = len(steps) - 1
        is_test = False

        full_loc[:steps[0],uid] = locs[0]
        for i in range(last_step):
            if locs[i] == 999999:
                full_loc[60*48:, uid] = 999999
                is_test = True
                break

            nextStep = min(steps[i+1], steps[i] + 24)
            full_loc[steps[i]: nextStep, uid] = locs[i]
            # full_loc[steps[i]:steps[i+1] , uid] = locs[i]

            # if nextStep - steps[i] == 1:
            #     continue

        # last one
        if is_test: continue
        nextStep =  min(steps[last_step] + 24, 3600)
        full_loc[steps[last_step]: nextStep, uid] = locs[last_step]
        # full_loc[steps[last_step]: , uid] = locs[last_step]


    full_df = pd.DataFrame(full_loc)
    full_df.columns = [i for i in range(max_uid)]

    for i in range(max_uid):
        most_common = full_df[~full_df[i].isin([-1,999999])][i].mode()[0]
        full_df.loc[full_df[i] == -1, i] = most_common
    full_df.transpose().to_csv('%s/city%s_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)

    return full_df

if __name__ == '__main__':
    city = sys.argv[1]
    filling_and_cal_conf(city)