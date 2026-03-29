"""
投资组合优化主引擎
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass, asdict

from barra.portfolio.config import (
    OPTIMIZATION_PARAMS, OUTPUT_CONFIG, DEFAULT_MARKET, DEFAULT_PORTFOLIO_VALUE
)
from barra.portfolio.data_loader import PortfolioDataLoader
from barra.portfolio.optimizer import QPOptimizer, OptimizationResult
from barra.portfolio.no_trade_zone import NoTradeZoneIterator, IterationResult, build_asset_covariance
from barra.portfolio.trade_generator import TradeGenerator
from barra.portfolio.output import PortfolioOutputManager
from utils import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


@dataclass
class PortfolioResult:
    """组合优化结果数据类"""
    trade_orders: pd.DataFrame
    position: pd.DataFrame
    active_risk: float
    turnover: float
    iterations: int
    converged: bool
    calc_date: str
    
    def to_dict(self) -> dict:
        """转换为字典（不含DataFrame）"""
        return {
            'active_risk': self.active_risk,
            'turnover': self.turnover,
            'iterations': self.iterations,
            'converged': self.converged,
            'calc_date': self.calc_date
        }


class PortfolioEngine:
    """投资组合优化主引擎
    
    编排完整的优化流程：
    1. 数据加载与对齐
    2. 构建资产协方差矩阵
    3. QP优化求解（可选）
    4. 无交易区域迭代
    5. 生成交易指令
    6. 保存结果
    
    使用示例：
        engine = PortfolioEngine(calc_date='2026-03-28')
        result = engine.run()
    """
    
    def __init__(
        self,
        calc_date: str,
        market: str = DEFAULT_MARKET,
        risk_output_dir: str = None,
        alpha_output_dir: str = None,
        output_dir: str = None,
        **optimization_params
    ):
        """初始化引擎
        
        Args:
            calc_date: 计算日期 'YYYY-MM-DD'
            market: 市场代码
            risk_output_dir: 风险模型输出目录
            alpha_output_dir: Alpha输出目录
            output_dir: 组合优化输出目录
            **optimization_params: 优化参数（覆盖默认配置）
        """
        self.calc_date = calc_date
        self.market = market
        
        # 合并优化参数
        self.params = OPTIMIZATION_PARAMS.copy()
        self.params.update(optimization_params)
        
        # 初始化组件
        self.data_loader = PortfolioDataLoader(
            market=market,
            risk_output_dir=risk_output_dir,
            alpha_output_dir=alpha_output_dir
        )
        self.output_manager = PortfolioOutputManager(output_dir=output_dir)
        self.trade_generator = TradeGenerator(
            min_trade_threshold=self.params['min_trade_threshold']
        )
        
        # 结果缓存
        self.data = None
        self.V = None
        self.iteration_result = None
        
    def run(
        self,
        position_input: Union[str, Dict, pd.Series] = 'zero',
        portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
        use_qp_init: bool = False,
        save_to_mysql: bool = False,
        portfolio_name: str = 'default'
    ) -> PortfolioResult:
        """执行完整优化流程
        
        Args:
            position_input: 当前持仓输入
            portfolio_value: 组合净值（元）
            use_qp_init: 是否用QP解作为迭代初始值
            save_to_mysql: 是否保存到MySQL
            portfolio_name: 组合名称
            
        Returns:
            PortfolioResult: 优化结果
        """
        logger.info('=' * 60)
        logger.info(f'投资组合优化开始: calc_date={self.calc_date}')
        logger.info('=' * 60)
        
        # Step 1: 加载并对齐数据
        logger.info('Step 1: 加载数据...')
        self.data = self.data_loader.align_all_data(
            calc_date=self.calc_date,
            position_input=position_input
        )
        
        # Step 2: 构建资产协方差矩阵
        logger.info('Step 2: 构建协方差矩阵...')
        self.V = build_asset_covariance(
            exposure=self.data['exposure'],
            factor_cov=self.data['factor_cov'],
            specific_risk=self.data['specific_risk']
        )
        logger.info('V shape: {}'.format(self.V.shape))
        
        # 准备numpy数组
        alpha = self.data['alpha'].values
        h_cur = self.data['current_position'].values - self.data['benchmark_weights'].values
        w_b = self.data['benchmark_weights'].values
        
        # Step 3: 初始解（可选）
        h_init = None
        if use_qp_init:
            logger.info('Step 3: QP优化求解初始解...')
            try:
                optimizer = QPOptimizer(
                    risk_aversion=self.params['risk_aversion'],
                    max_turnover=self.params['max_turnover'],
                    max_active_position=self.params['max_active_position']
                )
                qp_result = optimizer.solve(alpha, self.V, h_cur, w_b)
                h_init = qp_result.h_optimal
                logger.info(f'QP初始解: active_risk={qp_result.active_risk:.4f}')
            except Exception as e:
                logger.warning(f'QP优化失败，使用当前持仓作为初始值: {e}')
                h_init = h_cur.copy()
        else:
            logger.info('Step 3: 跳过QP优化，使用当前持仓作为初始值')
            h_init = h_cur.copy()
        
        # Step 4: 无交易区域迭代
        logger.info('Step 4: 无交易区域迭代...')
        iterator = NoTradeZoneIterator(
            risk_aversion=self.params['risk_aversion'],
            max_iterations=100,
            convergence_threshold=1e-6
        )
        self.iteration_result = iterator.iterate(alpha, self.V, h_cur, w_b, h_init)
        
        # Step 5: 生成交易指令
        logger.info('Step 5: 生成交易指令...')
        h_final = pd.Series(self.iteration_result.h_final, index=self.data['instruments'])
        h_cur_series = pd.Series(h_cur, index=self.data['instruments'])
        
        trade_orders = self.trade_generator.generate(
            h_final=h_final,
            h_cur=h_cur_series,
            portfolio_value=portfolio_value,
            prices=self.data['prices'],
            w_b=self.data['benchmark_weights']
        )
        
        # 生成持仓摘要
        position = self.trade_generator.generate_position_summary(trade_orders, portfolio_value)
        
        # 计算换手率
        turnover = np.sum(np.abs(self.iteration_result.h_final - h_cur))
        
        # Step 6: 保存结果
        logger.info('Step 6: 保存结果...')
        self.output_manager.save_trade_orders(trade_orders, self.calc_date)
        self.output_manager.save_position(position, self.calc_date)
        
        # 保存诊断日志
        v_diag = np.diag(self.V)
        self.output_manager.save_optimization_log(
            instruments=self.data['instruments'],
            alpha=self.data['alpha'],
            mcva=self.iteration_result.marginal_contributions,
            in_no_trade_zone=self.iteration_result.in_no_trade_zone,
            v_diag=v_diag,
            calc_date=self.calc_date
        )
        
        # 保存到MySQL
        if save_to_mysql:
            self.output_manager.save_to_mysql(
                trade_orders, position, self.calc_date, portfolio_name
            )
        
        # 构建结果
        result = PortfolioResult(
            trade_orders=trade_orders,
            position=position,
            active_risk=self.iteration_result.active_risk,
            turnover=turnover,
            iterations=self.iteration_result.iterations,
            converged=self.iteration_result.converged,
            calc_date=self.calc_date
        )
        
        # 打印摘要
        self.print_summary(result)
        
        logger.info('=' * 60)
        logger.info('投资组合优化完成')
        logger.info('=' * 60)
        
        return result
    
    def print_summary(self, result: PortfolioResult):
        """打印优化结果摘要
        
        Args:
            result: 优化结果
        """
        print('\n' + '=' * 60)
        print('投资组合优化结果摘要')
        print('=' * 60)
        print(f'计算日期: {result.calc_date}')
        print(f'主动风险: {result.active_risk:.4f} ({result.active_risk*100:.2f}%)')
        print(f'换手率: {result.turnover:.4f} ({result.turnover*100:.2f}%)')
        print(f'迭代次数: {result.iterations}')
        print(f'收敛状态: {"已收敛" if result.converged else "未收敛"}')
        print('-' * 60)
        
        # 交易统计
        orders = result.trade_orders
        buy_count = (orders['direction'] == 'buy').sum()
        sell_count = (orders['direction'] == 'sell').sum()
        hold_count = (orders['direction'] == 'hold').sum()
        
        print(f'买入股票: {buy_count}只')
        print(f'卖出股票: {sell_count}只')
        print(f'持有股票: {hold_count}只')
        print('-' * 60)
        
        # 约束检查
        h_final = self.iteration_result.h_final
        w_b = self.data['benchmark_weights'].values
        
        cash_neutral = abs(np.sum(h_final))
        short_violation = (h_final < -w_b).sum()
        
        print(f'现金中性偏差: {cash_neutral:.2e}')
        print(f'卖空约束违反: {short_violation}只')
        print('=' * 60)
