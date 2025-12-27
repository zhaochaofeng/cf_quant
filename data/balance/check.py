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
    'report_type': 'report_type',
    'comp_type': 'comp_type',
    'end_type': 'end_type',
    'total_share': 'total_share',
    'cap_rese': 'cap_rese',
    'undistr_porfit': 'undistr_porfit',
    'surplus_rese': 'surplus_rese',
    'special_rese': 'special_rese',
    'money_cap': 'money_cap',
    'trad_asset': 'trad_asset',
    'notes_receiv': 'notes_receiv',
    'accounts_receiv': 'accounts_receiv',
    'oth_receiv': 'oth_receiv',
    'prepayment': 'prepayment',
    'div_receiv': 'div_receiv',
    'int_receiv': 'int_receiv',
    'inventories': 'inventories',
    'amor_exp': 'amor_exp',
    'nca_within_1y': 'nca_within_1y',
    'sett_rsrv': 'sett_rsrv',
    'loanto_oth_bank_fi': 'loanto_oth_bank_fi',
    'premium_receiv': 'premium_receiv',
    'reinsur_receiv': 'reinsur_receiv',
    'reinsur_res_receiv': 'reinsur_res_receiv',
    'pur_resale_fa': 'pur_resale_fa',
    'oth_cur_assets': 'oth_cur_assets',
    'total_cur_assets': 'total_cur_assets',
    'fa_avail_for_sale': 'fa_avail_for_sale',
    'htm_invest': 'htm_invest',
    'lt_eqt_invest': 'lt_eqt_invest',
    'invest_real_estate': 'invest_real_estate',
    'time_deposits': 'time_deposits',
    'oth_assets': 'oth_assets',
    'lt_rec': 'lt_rec',
    'fix_assets': 'fix_assets',
    'cip': 'cip',
    'const_materials': 'const_materials',
    'fixed_assets_disp': 'fixed_assets_disp',
    'produc_bio_assets': 'produc_bio_assets',
    'oil_and_gas_assets': 'oil_and_gas_assets',
    'intan_assets': 'intan_assets',
    'r_and_d': 'r_and_d',
    'goodwill': 'goodwill',
    'lt_amor_exp': 'lt_amor_exp',
    'defer_tax_assets': 'defer_tax_assets',
    'decr_in_disbur': 'decr_in_disbur',
    'oth_nca': 'oth_nca',
    'total_nca': 'total_nca',
    'cash_reser_cb': 'cash_reser_cb',
    'depos_in_oth_bfi': 'depos_in_oth_bfi',
    'prec_metals': 'prec_metals',
    'deriv_assets': 'deriv_assets',
    'rr_reins_une_prem': 'rr_reins_une_prem',
    'rr_reins_outstd_cla': 'rr_reins_outstd_cla',
    'rr_reins_lins_liab': 'rr_reins_lins_liab',
    'rr_reins_lthins_liab': 'rr_reins_lthins_liab',
    'refund_depos': 'refund_depos',
    'ph_pledge_loans': 'ph_pledge_loans',
    'refund_cap_depos': 'refund_cap_depos',
    'indep_acct_assets': 'indep_acct_assets',
    'client_depos': 'client_depos',
    'client_prov': 'client_prov',
    'transac_seat_fee': 'transac_seat_fee',
    'invest_as_receiv': 'invest_as_receiv',
    'total_assets': 'total_assets',
    'lt_borr': 'lt_borr',
    'st_borr': 'st_borr',
    'cb_borr': 'cb_borr',
    'depos_ib_deposits': 'depos_ib_deposits',
    'loan_oth_bank': 'loan_oth_bank',
    'trading_fl': 'trading_fl',
    'notes_payable': 'notes_payable',
    'acct_payable': 'acct_payable',
    'adv_receipts': 'adv_receipts',
    'sold_for_repur_fa': 'sold_for_repur_fa',
    'comm_payable': 'comm_payable',
    'payroll_payable': 'payroll_payable',
    'taxes_payable': 'taxes_payable',
    'int_payable': 'int_payable',
    'div_payable': 'div_payable',
    'oth_payable': 'oth_payable',
    'acc_exp': 'acc_exp',
    'deferred_inc': 'deferred_inc',
    'st_bonds_payable': 'st_bonds_payable',
    'payable_to_reinsurer': 'payable_to_reinsurer',
    'rsrv_insur_cont': 'rsrv_insur_cont',
    'acting_trading_sec': 'acting_trading_sec',
    'acting_uw_sec': 'acting_uw_sec',
    'non_cur_liab_due_1y': 'non_cur_liab_due_1y',
    'oth_cur_liab': 'oth_cur_liab',
    'total_cur_liab': 'total_cur_liab',
    'bond_payable': 'bond_payable',
    'lt_payable': 'lt_payable',
    'specific_payables': 'specific_payables',
    'estimated_liab': 'estimated_liab',
    'defer_tax_liab': 'defer_tax_liab',
    'defer_inc_non_cur_liab': 'defer_inc_non_cur_liab',
    'oth_ncl': 'oth_ncl',
    'total_ncl': 'total_ncl',
    'depos_oth_bfi': 'depos_oth_bfi',
    'deriv_liab': 'deriv_liab',
    'depos': 'depos',
    'agency_bus_liab': 'agency_bus_liab',
    'oth_liab': 'oth_liab',
    'prem_receiv_adva': 'prem_receiv_adva',
    'depos_received': 'depos_received',
    'ph_invest': 'ph_invest',
    'reser_une_prem': 'reser_une_prem',
    'reser_outstd_claims': 'reser_outstd_claims',
    'reser_lins_liab': 'reser_lins_liab',
    'reser_lthins_liab': 'reser_lthins_liab',
    'indept_acc_liab': 'indept_acc_liab',
    'pledge_borr': 'pledge_borr',
    'indem_payable': 'indem_payable',
    'policy_div_payable': 'policy_div_payable',
    'total_liab': 'total_liab',
    'treasury_share': 'treasury_share',
    'ordin_risk_reser': 'ordin_risk_reser',
    'forex_differ': 'forex_differ',
    'invest_loss_unconf': 'invest_loss_unconf',
    'minority_int': 'minority_int',
    'total_hldr_eqy_exc_min_int': 'total_hldr_eqy_exc_min_int',
    'total_hldr_eqy_inc_min_int': 'total_hldr_eqy_inc_min_int',
    'total_liab_hldr_eqy': 'total_liab_hldr_eqy',
    'lt_payroll_payable': 'lt_payroll_payable',
    'oth_comp_income': 'oth_comp_income',
    'oth_eqt_tools': 'oth_eqt_tools',
    'oth_eqt_tools_p_shr': 'oth_eqt_tools_p_shr',
    'lending_funds': 'lending_funds',
    'acc_receivable': 'acc_receivable',
    'st_fin_payable': 'st_fin_payable',
    'payables': 'payables',
    'hfs_assets': 'hfs_assets',
    'hfs_sales': 'hfs_sales',
    'cost_fin_assets': 'cost_fin_assets',
    'fair_value_fin_assets': 'fair_value_fin_assets',
    'contract_assets': 'contract_assets',
    'contract_liab': 'contract_liab',
    'accounts_receiv_bill': 'accounts_receiv_bill',
    'accounts_pay': 'accounts_pay',
    'oth_rcv_total': 'oth_rcv_total',
    'fix_assets_total': 'fix_assets_total',
    'cip_total': 'cip_total',
    'oth_pay_total': 'oth_pay_total',
    'long_pay_total': 'long_pay_total',
    'debt_invest': 'debt_invest',
    'oth_debt_invest': 'oth_debt_invest'
}


def main(start_date: str, end_date: str):
    try:
        t = time.time()
        check = CheckMySQLData(
            start_date=start_date,
            end_date=end_date,
            table_name='balance_ts',
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
                                         api_fun='balancesheet',
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
            send_email('Data:Check:balance_ts', '\n'.join(res))
        print('耗时：{}s'.format(round(time.time() - t, 4)))
    except Exception as e:
        error_msg = traceback.format_exc()
        send_email('Data:Check:balance_ts', error_msg)


if __name__ == '__main__':
    fire.Fire(main)
