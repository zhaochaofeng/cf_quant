import logging
import warnings
from pprint import pprint

import pandas as pd

from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import indicator_analysis
from qlib.contrib.evaluate import risk_analysis
from qlib.log import get_module_logger
from qlib.utils import fill_placeholder
from qlib.utils import flatten_dict
from qlib.utils import get_date_by_shift
from qlib.workflow.record_temp import PortAnaRecord

logger = get_module_logger("workflow", logging.INFO)


class RollingPortAnaRecord(PortAnaRecord):
    """
    Rolling 训练的投资组合分析
    """

    artifact_path = "portfolio_analysis"
    depend_cls = None

    def __init__(self, recorder, rolling_pred_df, **kwargs):
        """
        Args:
            recorder: 记录器对象
            rolling_pred_df: 合并后的滚动预测结果DataFrame
        """
        super().__init__(recorder, **kwargs)
        self.rolling_pred_df = rolling_pred_df  # 传入合并后的预测数据

    def _generate(self, **kwargs):
        # 加载预测数据
        pred = self.rolling_pred_df

        # replace the "<PRED>" with prediction saved before
        placeholder_value = {"<PRED>": pred}
        for k in "executor_config", "strategy_config":
            setattr(self, k, fill_placeholder(getattr(self, k), placeholder_value))

        # if the backtesting time range is not set, it will automatically extract time range from the prediction file
        dt_values = pred.index.get_level_values("datetime")
        if self.backtest_config["start_time"] is None:
            self.backtest_config["start_time"] = dt_values.min()
        if self.backtest_config["end_time"] is None:
            self.backtest_config["end_time"] = get_date_by_shift(dt_values.max(), 1)

        artifact_objects = {}
        # custom strategy and get backtest
        # normal_backtest -> qlib.backtest.batcktest
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=self.executor_config, strategy=self.strategy_config, **self.backtest_config
        )
        # report_normal: 收益数据
        # positions_normal: 持仓数据
        for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
            artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})
        # indicator_normal: 指标数据
        for _freq, indicators_normal in indicator_dict.items():
            artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
            artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})

        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq not in portfolio_metric_dict:
                warnings.warn(
                    f"the freq {_analysis_freq} report is not found, please set the corresponding env with `generate_portfolio_metrics=True`"
                )
            else:
                # risk_analysis
                report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"], freq=_analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_analysis_freq
                )

                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                # log metrics
                analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})
                logger.info(
                    f"Portfolio analysis record 'port_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                # print out results
                pprint(f"The following are analysis results of benchmark return({_analysis_freq}).")
                pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
                pprint(f"The following are analysis results of the excess return without cost({_analysis_freq}).")
                pprint(analysis["excess_return_without_cost"])
                pprint(f"The following are analysis results of the excess return with cost({_analysis_freq}).")
                pprint(analysis["excess_return_with_cost"])

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq not in indicator_dict:
                warnings.warn(f"the freq {_analysis_freq} indicator is not found")
            else:
                indicators_normal = indicator_dict.get(_analysis_freq)[0]
                if self.indicator_analysis_method is None:
                    # 指标计算
                    analysis_df = indicator_analysis(indicators_normal)
                else:
                    analysis_df = indicator_analysis(indicators_normal, method=self.indicator_analysis_method)
                # log metrics
                analysis_dict = analysis_df["value"].to_dict()
                self.recorder.log_metrics(**{f"{_analysis_freq}.{k}": v for k, v in analysis_dict.items()})
                # save results
                artifact_objects.update({f"indicator_analysis_{_analysis_freq}.pkl": analysis_df})
                logger.info(
                    f"Indicator analysis record 'indicator_analysis_{_analysis_freq}.pkl' has been saved as the artifact of the Experiment {self.recorder.experiment_id}"
                )
                pprint(f"The following are analysis results of indicators({_analysis_freq}).")
                pprint(analysis_df)
        return artifact_objects



