'''
    Tushare利润表入库MySQL
'''
import time
import fire
import traceback
from utils import send_email
from data.process_data import TSFinacialData

feas = {
    'ts_code': 'ts_code',
    'qlib_code': 'ts_code',
    'ann_date': 'ann_date',
    'f_ann_date': 'f_ann_date',
    'end_date': 'end_date',
    'report_type': 'report_type',
    'comp_type': 'comp_type',
    'end_type': 'end_type',
    'basic_eps': 'basic_eps',
    'diluted_eps': 'diluted_eps',
    'total_revenue': 'total_revenue',
    'revenue': 'revenue',
    'int_income': 'int_income',
    'prem_earned': 'prem_earned',
    'comm_income': 'comm_income',
    'n_commis_income': 'n_commis_income',
    'n_oth_income': 'n_oth_income',
    'n_oth_b_income': 'n_oth_b_income',
    'prem_income': 'prem_income',
    'out_prem': 'out_prem',
    'une_prem_reser': 'une_prem_reser',
    'reins_income': 'reins_income',
    'n_sec_tb_income': 'n_sec_tb_income',
    'n_sec_uw_income': 'n_sec_uw_income',
    'n_asset_mg_income': 'n_asset_mg_income',
    'oth_b_income': 'oth_b_income',
    'fv_value_chg_gain': 'fv_value_chg_gain',
    'invest_income': 'invest_income',
    'ass_invest_income': 'ass_invest_income',
    'forex_gain': 'forex_gain',
    'total_cogs': 'total_cogs',
    'oper_cost': 'oper_cost',
    'int_exp': 'int_exp',
    'comm_exp': 'comm_exp',
    'biz_tax_surchg': 'biz_tax_surchg',
    'sell_exp': 'sell_exp',
    'admin_exp': 'admin_exp',
    'fin_exp': 'fin_exp',
    'assets_impair_loss': 'assets_impair_loss',
    'prem_refund': 'prem_refund',
    'compens_payout': 'compens_payout',
    'reser_insur_liab': 'reser_insur_liab',
    'div_payt': 'div_payt',
    'reins_exp': 'reins_exp',
    'oper_exp': 'oper_exp',
    'compens_payout_refu': 'compens_payout_refu',
    'insur_reser_refu': 'insur_reser_refu',
    'reins_cost_refund': 'reins_cost_refund',
    'other_bus_cost': 'other_bus_cost',
    'operate_profit': 'operate_profit',
    'non_oper_income': 'non_oper_income',
    'non_oper_exp': 'non_oper_exp',
    'nca_disploss': 'nca_disploss',
    'total_profit': 'total_profit',
    'income_tax': 'income_tax',
    'n_income': 'n_income',
    'n_income_attr_p': 'n_income_attr_p',
    'minority_gain': 'minority_gain',
    'oth_compr_income': 'oth_compr_income',
    't_compr_income': 't_compr_income',
    'compr_inc_attr_p': 'compr_inc_attr_p',
    'compr_inc_attr_m_s': 'compr_inc_attr_m_s',
    'ebit': 'ebit',
    'ebitda': 'ebitda',
    'insurance_exp': 'insurance_exp',
    'undist_profit': 'undist_profit',
    'distable_profit': 'distable_profit',
    'rd_exp': 'rd_exp',
    'fin_exp_int_exp': 'fin_exp_int_exp',
    'fin_exp_int_inc': 'fin_exp_int_inc',
    'transfer_surplus_rese': 'transfer_surplus_rese',
    'transfer_housing_imprest': 'transfer_housing_imprest',
    'transfer_oth': 'transfer_oth',
    'adj_lossgain': 'adj_lossgain',
    'withdra_legal_surplus': 'withdra_legal_surplus',
    'withdra_legal_pubfund': 'withdra_legal_pubfund',
    'withdra_biz_devfund': 'withdra_biz_devfund',
    'withdra_rese_fund': 'withdra_rese_fund',
    'withdra_oth_ersu': 'withdra_oth_ersu',
    'workers_welfare': 'workers_welfare',
    'distr_profit_shrhder': 'distr_profit_shrhder',
    'prfshare_payable_dvd': 'prfshare_payable_dvd',
    'comshare_payable_dvd': 'comshare_payable_dvd',
    'capit_comstock_div': 'capit_comstock_div',
    'continued_net_profit': 'continued_net_profit',
    'update_flag': 'update_flag'
}

def main(
        start_date: str,
        end_date: str,
        now_date: str = None
        ):
    try:
        t = time.time()
        processor = TSFinacialData(
            start_date,
            end_date,
            now_date,
            feas=feas,
            table_name='income_ts'
        )
        stocks = processor.get_stocks()
        df = processor.fetch_data_from_api(stocks[0:10])
        data = processor.process(df)
        processor.write_to_mysql(data)
        processor.logger.info('耗时： s'.format(time.time() - t, 4))
    except:
        err_msg = traceback.format_exc()
        send_email('Data: income_ts', err_msg)

if __name__ == '__main__':
    fire.Fire(main)

