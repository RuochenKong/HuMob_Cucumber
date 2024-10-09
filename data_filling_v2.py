import pandas as pd
import numpy as np
import os
import sys

outdir = 'filled_data/v3'
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
        full_loc[0 : steps[0], uid] = locs[0]
        for i in range(last_step):
            if locs[i] == 999999:
                full_loc[60*48:, uid] = 999999
                is_test = True
                break

            num_sep = steps[i+1] - steps[i]
            if num_sep == 1:
                full_loc[steps[i], uid] = locs[i]
                continue

            num_sep = steps[i+1] - steps[i]
            if num_sep == 1:
                full_loc[steps[i], uid] = locs[i]
                continue
            mid_point = steps[i]+int((num_sep+1)/2)
            full_loc[steps[i]: mid_point, uid] = locs[i]
            full_loc[mid_point : steps[i+1], uid] = locs[i+1]


        # last one
        if is_test: continue

        full_loc[steps[last_step]:, uid] = locs[last_step]

    full_df = pd.DataFrame(full_loc)
    full_df.columns = [i for i in range(max_uid)]

    most_common = full_df[~full_df.isin([-1,999999])].mode()[0]
    for i in range(max_uid):
        full_df.loc[full_df[i] == -1, i] = most_common
    full_df.transpose().to_csv('%s/city%s_data.csv.gz'%(outdir,cityname), compression='gzip', index=False)

    return full_df

if __name__ == '__main__':
    city = sys.argv[1]
    filling_and_cal_conf(city)