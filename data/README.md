
## 财务数据
1.表 income_ts, cashflow_ts, balance_ts 表中包含少量（0.8%-2.3%）update_flag=0 and ann_date!=f_ann_date 数据。
```
    select sum(a.flag), count(a.flag) from
(select 
	case when update_flag=0 and ann_date!=f_ann_date then 1 
	else 0 end flag
from income_ts )a;
```
2. 存在 ts_code, end_date, ann_date, f_ann_date 字段值相同的数据，也就是同一天做了数据修改，此时需要使用update_flag=1 的数据






