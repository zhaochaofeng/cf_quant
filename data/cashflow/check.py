'''
    检查一段时间 MySQL 与 Tushare 数据是否一致
'''

import time
import fire
import traceback
import pandas as pd
from utils import send_email
from data.check_data import CheckMySQLData


feas = {
    'ts_code': 'ts_code',
    'end_date': 'end_date',
    'f_ann_date': 'f_ann_date',
    'update_flag': 'update_flag',
    'ann_date': 'ann_date',
    'comp_type': 'comp_type',
    'report_type': 'report_type',
    'end_type': 'end_type',
    'net_profit': 'net_profit',
    'finan_exp': 'finan_exp',
    'c_fr_sale_sg': 'c_fr_sale_sg',
    'recp_tax_rends': 'recp_tax_rends',
    'n_depos_incr_fi': 'n_depos_incr_fi',
    'n_incr_loans_cb': 'n_incr_loans_cb',
    'n_inc_borr_oth_fi': 'n_inc_borr_oth_fi',
    'prem_fr_orig_contr': 'prem_fr_orig_contr',
    'n_incr_insured_dep': 'n_incr_insured_dep',
    'n_reinsur_prem': 'n_reinsur_prem',
    'n_incr_disp_tfa': 'n_incr_disp_tfa',
    'ifc_cash_incr': 'ifc_cash_incr',
    'n_incr_disp_faas': 'n_incr_disp_faas',
    'n_incr_loans_oth_bank': 'n_incr_loans_oth_bank',
    'n_cap_incr_repur': 'n_cap_incr_repur',
    'c_fr_oth_operate_a': 'c_fr_oth_operate_a',
    'c_inf_fr_operate_a': 'c_inf_fr_operate_a',
    'c_paid_goods_s': 'c_paid_goods_s',
    'c_paid_to_for_empl': 'c_paid_to_for_empl',
    'c_paid_for_taxes': 'c_paid_for_taxes',
    'n_incr_clt_loan_adv': 'n_incr_clt_loan_adv',
    'n_incr_dep_cbob': 'n_incr_dep_cbob',
    'c_pay_claims_orig_inco': 'c_pay_claims_orig_inco',
    'pay_handling_chrg': 'pay_handling_chrg',
    'pay_comm_insur_plcy': 'pay_comm_insur_plcy',
    'oth_cash_pay_oper_act': 'oth_cash_pay_oper_act',
    'st_cash_out_act': 'st_cash_out_act',
    'n_cashflow_act': 'n_cashflow_act',
    'oth_recp_ral_inv_act': 'oth_recp_ral_inv_act',
    'c_disp_withdrwl_invest': 'c_disp_withdrwl_invest',
    'c_recp_return_invest': 'c_recp_return_invest',
    'n_recp_disp_fiolta': 'n_recp_disp_fiolta',
    'n_recp_disp_sobu': 'n_recp_disp_sobu',
    'stot_inflows_inv_act': 'stot_inflows_inv_act',
    'c_pay_acq_const_fiolta': 'c_pay_acq_const_fiolta',
    'c_paid_invest': 'c_paid_invest',
    'n_disp_subs_oth_biz': 'n_disp_subs_oth_biz',
    'oth_pay_ral_inv_act': 'oth_pay_ral_inv_act',
    'n_incr_pledge_loan': 'n_incr_pledge_loan',
    'stot_out_inv_act': 'stot_out_inv_act',
    'n_cashflow_inv_act': 'n_cashflow_inv_act',
    'c_recp_borrow': 'c_recp_borrow',
    'proc_issue_bonds': 'proc_issue_bonds',
    'oth_cash_recp_ral_fnc_act': 'oth_cash_recp_ral_fnc_act',
    'stot_cash_in_fnc_act': 'stot_cash_in_fnc_act',
    'free_cashflow': 'free_cashflow',
    'c_prepay_amt_borr': 'c_prepay_amt_borr',
    'c_pay_dist_dpcp_int_exp': 'c_pay_dist_dpcp_int_exp',
    'incl_dvd_profit_paid_sc_ms': 'incl_dvd_profit_paid_sc_ms',
    'oth_cashpay_ral_fnc_act': 'oth_cashpay_ral_fnc_act',
    'stot_cashout_fnc_act': 'stot_cashout_fnc_act',
    'n_cash_flows_fnc_act': 'n_cash_flows_fnc_act',
    'eff_fx_flu_cash': 'eff_fx_flu_cash',
    'n_incr_cash_cash_equ': 'n_incr_cash_cash_equ',
    'c_cash_equ_beg_period': 'c_cash_equ_beg_period',
    'c_cash_equ_end_period': 'c_cash_equ_end_period',
    'c_recp_cap_contrib': 'c_recp_cap_contrib',
    'incl_cash_rec_saims': 'incl_cash_rec_saims',
    'uncon_invest_loss': 'uncon_invest_loss',
    'prov_depr_assets': 'prov_depr_assets',
    'depr_fa_coga_dpba': 'depr_fa_coga_dpba',
    'amort_intang_assets': 'amort_intang_assets',
    'lt_amort_deferred_exp': 'lt_amort_deferred_exp',
    'decr_deferred_exp': 'decr_deferred_exp',
    'incr_acc_exp': 'incr_acc_exp',
    'loss_disp_fiolta': 'loss_disp_fiolta',
    'loss_scr_fa': 'loss_scr_fa',
    'loss_fv_chg': 'loss_fv_chg',
    'invest_loss': 'invest_loss',
    'decr_def_inc_tax_assets': 'decr_def_inc_tax_assets',
    'incr_def_inc_tax_liab': 'incr_def_inc_tax_liab',
    'decr_inventories': 'decr_inventories',
    'decr_oper_payable': 'decr_oper_payable',
    'incr_oper_payable': 'incr_oper_payable',
    'others': 'others',
    'im_net_cashflow_oper_act': 'im_net_cashflow_oper_act',
    'conv_debt_into_cap': 'conv_debt_into_cap',
    'conv_copbonds_due_within_1y': 'conv_copbonds_due_within_1y',
    'fa_fnc_leases': 'fa_fnc_leases',
    'im_n_incr_cash_equ': 'im_n_incr_cash_equ',
    'net_dism_capital_add': 'net_dism_capital_add',
    'net_cash_rece_sec': 'net_cash_rece_sec',
    'credit_impa_loss': 'credit_impa_loss',
    'use_right_asset_dep': 'use_right_asset_dep',
    'oth_loss_asset': 'oth_loss_asset',
    'end_bal_cash': 'end_bal_cash',
    'beg_bal_cash': 'beg_bal_cash',
    'end_bal_cash_equ': 'end_bal_cash_equ',
    'beg_bal_cash_equ': 'beg_bal_cash_equ'
}


def main(start_date: str, end_date: str):
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='cashflow_ts',
            feas=list(feas.values())
        )
        # [start_date, end_date] 需要设置超过1个季度，否则可能出现stocks 为空
        sql = f"""
            SELECT {','.join(list(feas.values()))} FROM {check.table_name} 
            WHERE ann_date>='{check.start_date}' AND ann_date<='{check.end_date}'
        """
        df_mysql = check.fetch_data_from_mysql(sql_str=sql, is_fin=True)
        if df_mysql is None or df_mysql.empty:
            raise ValueError('mysql data is empty')
        # test
        # stocks = df_mysql['ts_code'].unique().tolist()
        # df_mysql = df_mysql[df_mysql['ts_code'].isin(stocks[0:50])]
        df_mysql.set_index(list(feas.values())[0:4], inplace=True)
        df_mysql.sort_index(axis=0, inplace=True)

        stocks = df_mysql.index.get_level_values('ts_code').unique().tolist()

        df_ts = check.fetch_data_from_ts(stocks,
                                         api_fun='cashflow',
                                         batch_size=1,
                                         req_per_min=500,
                                         feas=list(feas.keys()),
                                         is_fin=True)
        if df_ts is None or df_ts.empty:
            raise ValueError('tushare data is empty')
        df_ts.rename(columns=feas, inplace=True)
        for date in ['end_date', 'ann_date', 'f_ann_date']:
            df_ts[date] = pd.to_datetime(df_ts[date]).dt.date
        for col in ['update_flag', 'report_type', 'comp_type']:
            df_ts[col] = df_ts[col].astype('int64', errors='ignore')
        df_ts['end_type'] = df_ts['end_type'].astype('float32')

        df_ts.set_index(list(feas.values())[0:4], inplace=True)
        df_ts.sort_index(axis=0, inplace=True)
        res = check.check(df_mysql, df_ts, is_repair=True, idx_num=4)
        if len(res) != 0:
            send_email('Data:Check:cashflow_ts (Auto repair)', '\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:cashflow_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
