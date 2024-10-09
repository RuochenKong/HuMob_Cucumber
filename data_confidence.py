import pandas as pd
import numpy as np
import os
import sys

if not os.path.exists('confidence_data'):
    os.mkdir('confidence_data')

def filling_and_cal_conf(cityname, denum,xpow):
    df = pd.read_csv('processed_data/city%s_data.csv.gz'%cityname, compression='gzip')
    max_uid = df['uid'].max() + 1
    confidence_np = np.ones((75*48,max_uid))
    for uid,udf in df.groupby('uid'):
        steps = list(udf['step'])
        locs = list(udf['loc'])
        last_step = len(steps) - 1
        confidence_np[0:steps[0],uid] = 0
        is_test = False
        for i in range(last_step):
            if locs[i] == 999999:
                confidence_np[60*48:, uid] = -1
                is_test = True
                break

            if steps[i+1] - steps[i] == 1:
                continue

            conf = np.array([i for i in range(steps[i+1] - steps[i])])
            conf = np.power((conf/denum),xpow)
            conf = np.exp(-conf)
            confidence_np[steps[i]:steps[i+1],uid] = conf
        # last one
        if is_test: continue

        conf = np.array([i for i in range(3600-steps[last_step])])
        conf = np.power((conf/denum),xpow)
        conf = np.exp(-conf)
        confidence_np[steps[last_step]:3600,uid] = conf

    conf_df = pd.DataFrame(confidence_np)
    conf_df.columns = [i for i in range(max_uid)]

    conf_df.transpose().to_csv('confidence_data/city%s_confidence_%d_%.1f.csv.gz'%(cityname,denum,xpow), compression='gzip', index=False)

if __name__ == '__main__':
    city = sys.argv[1]
    denum = int(sys.argv[2])
    xpow = float(sys.argv[3])
    filling_and_cal_conf(city,denum,xpow)