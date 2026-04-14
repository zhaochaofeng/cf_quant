'''
    修改csi300.txt, csi500.txt 第3列的日期为当天
'''

import fire
import os
import traceback
from datetime import datetime

import pandas as pd

from utils import send_email


def change(path: str = None):
    path = os.path.expanduser(path)
    print('path: {}'.format(path))

    date = datetime.now().strftime("%Y-%m-%d")

    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['code', 'start_date', 'end_date']
    # 仅修改当前存在指数中的成分股 end_date
    old_date = df['end_date'].max()
    if old_date == date:
        print('No need to change, old_date: {}'.format(old_date))
        return
    print('old_date: {}'.format(old_date))
    print(df.head())

    df.loc[df['end_date'] == old_date, 'end_date'] = date
    path_new = path + '.tmp'
    df.to_csv(path_new, sep='\t', header=False, index=False)

    if os.path.getsize(path_new) > 0:
        os.remove(path)
        os.rename(path_new, path)


def main(
        path='~/.qlib/qlib_data/index/instruments',
        csi_list: list = None
):
    try:
        if csi_list is None:
            csi_list = ['csi300.txt', 'csi500.txt']
        for csi in csi_list:
            change(path=os.path.join(path, csi))
    except:
        err_msg = traceback.format_exc()
        send_email('Data: index: change_end_date', err_msg)


if __name__ == '__main__':
    fire.Fire(main)
