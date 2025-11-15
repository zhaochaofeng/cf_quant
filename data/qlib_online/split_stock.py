"""
    将 股票数 写到单个文件
"""
import os
import fire
import time
import traceback
import shutil
import pandas as pd
from utils import send_email


def main(path_in: str, path_out: str, start_date: str, end_date: str):
    print('\n{}\n{}'.format('=' * 100, 'split_stock ...'))
    try:
        t = time.time()
        df = pd.read_csv(path_in, sep='\t')
        columns = df.columns.tolist()

        output_dir = os.path.join(path_out, 'pit_{}_{}'.format(start_date, end_date))
        output_dir = os.path.expanduser(output_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 按股票代码分组并保存
        for i, (code, group) in enumerate(df.groupby('symbol')):
            output_file = os.path.join(output_dir, f"{code}.csv")
            group.to_csv(output_file, index=False, columns=columns)
            if (i + 1) % 100 == 0:
                print(f"processed : {i + 1} ")
        print('耗时：{} s'.format(round(time.time() - t, 4)))
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
        send_email('Data:trade_daily:split_stock', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
