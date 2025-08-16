from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import logging

import qlib
from qlib.workflow.expm import MLflowExpManager
from qlib.utils import init_instance_by_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="QLib量化模型预测服务")

# 定义请求数据模型
class PredictionRequest(BaseModel):
    stock_codes: List[str]  # 股票代码列表，如 ["SH600000", "SZ000001"]
    start_date: str  # 开始日期，格式如 "2023-01-01"
    end_date: Optional[str] = None  # 结束日期，可选，默认为开始日期


# 定义响应数据模型
class PredictionResponse(BaseModel):
    request_id: str
    predictions: Dict[str, float]
    status: str = "success"

# QLib模型加载器
class QLibModelLoader:
    def __init__(self, provider_uri: str, uri: str, exp_id: str):
        self.provider_uri = provider_uri   # 数据地址
        self.uri = uri                     # 模型地址
        self.exp_id = exp_id               # 实验id
        self.model = None
        self.initialized = False
        self.task_config = None
        self.recorder = None

    def initialize(self):
        """初始化QLib和加载模型"""
        try:
            # 初始化QLib，使用默认配置
            qlib.init(provider_uri=self.provider_uri)
            logger.info("QLib初始化成功")

            # 加载实验记录
            exp_manager = MLflowExpManager(uri=self.uri, default_exp_name='default_exp')
            exp = exp_manager.get_exp(experiment_id=self.exp_id)
            if not exp:
                raise ValueError(f"找不到ID为{self.exp_id}的实验记录")

            # 获取最新的在线模型记录器
            recorders = exp.list_recorders()
            online_recorders = []
            for rid, rec in recorders.items():
                if rec.status != 'FINISHED':
                    continue
                tags = rec.list_tags()
                if tags.get('online_status') == 'online':
                    online_recorders.append(rec)

            if not online_recorders:
                raise ValueError(f"实验{self.exp_id}中没有找到在线模型记录")

            # 选择最新的在线模型
            newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)
            self.recorder = newest_recorder

            # 加载模型
            self.model = self.recorder.load_object("params.pkl")
            # 记载配置文件
            self.task_config = self.recorder.load_object("task")
            self.initialized = True
            logger.info(f"模型加载成功，实验ID: {self.exp_id}")
            return True
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            self.initialized = False
            return False

    def reload_if_needed(self):
        """检查是否有新的在线模型并重新加载"""
        try:
            if not self.recorder:
                return self.initialize()

            # 重新获取实验
            exp_manager = MLflowExpManager(uri=self.uri, default_exp_name='default_exp')
            exp = exp_manager.get_exp(experiment_id=self.exp_id)

            # 获取当前在线模型
            recorders = exp.list_recorders()
            online_recorders = []
            for rid, rec in recorders.items():
                if rec.status != 'FINISHED':
                    continue
                tags = rec.list_tags()
                if tags.get('online_status') == 'online':
                    online_recorders.append(rec)

            if not online_recorders:
                logger.warning("没有找到在线模型")
                return False

            # 选择最新的在线模型
            newest_recorder = max(online_recorders, key=lambda rec: rec.start_time)

            # 如果是新模型，则重新加载
            if newest_recorder.id != self.recorder.id:
                logger.info(f"检测到新在线模型 {newest_recorder.id}，正在重新加载...")
                self.recorder = newest_recorder
                self.model = self.recorder.load_object("params.pkl")
                self.task_config = self.recorder.load_object("task")
                logger.info(f"新模型加载成功，记录器ID: {self.recorder.id}")
                return True
            return False
        except Exception as e:
            logger.error(f"模型重新加载失败: {str(e)}")
            return False

    def predict(self, stock_codes: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用模型进行预测"""
        if not self.initialized or self.model is None:
            raise RuntimeError("模型未初始化，请先调用initialize()")

        # 检查是否需要重新加载模型
        self.reload_if_needed()

        dataset_config = self.task_config["dataset"]
        dataset_config['kwargs']['handler']['kwargs']['end_time'] = max(
            end_date, dataset_config['kwargs']['handler']['kwargs']['end_time'])  # 全局日期
        if len(stock_codes) > 0:
            dataset_config['kwargs']['handler']['kwargs']['instruments'] = stock_codes  # 修改股票集合
        dataset_config['kwargs']['segments']['test'] = (pd.Timestamp(start_date), pd.Timestamp(end_date))  # 修改测试

        try:
            # 数据集
            dataset = init_instance_by_config(dataset_config)
            # 预测。Series
            predictions = self.model.predict(dataset, segment='test')
            # 转化为DataFrame
            predictions = predictions.to_frame('score')
            # 按照datetime,instrument排序
            predictions = predictions.sort_index()
            return predictions
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            raise

# 全局模型加载器实例
# 请替换为你的实际实验ID
provider_uri = '~/.qlib/qlib_data/custom_data_hfq'
uri = '/Users/chaofeng/code/cf_quant/strategy/lightGBM/mlruns'
exp_id = '475678663686452018'
model_loader = QLibModelLoader(provider_uri, uri, exp_id)

# 应用启动时加载模型
@app.on_event("startup")
async def startup_event():
    if not model_loader.initialize():
        raise RuntimeError("模型加载失败，服务无法启动")


# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_loader.initialized else "unhealthy",
        "model_loaded": model_loader.initialized,
        "experiment_id": model_loader.exp_id
    }


# 预测接口
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 生成简单的请求ID
        request_id = f"req_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

        # 调用模型进行预测
        predictions = model_loader.predict(
            stock_codes=request.stock_codes,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # 整理预测结果
        result = {}
        for index, row in predictions.reset_index().iterrows():
            result[row['instrument']] = row['score']
        return {
            "request_id": request_id,
            "predictions": result
        }

    except Exception as e:
        logger.error(f"预测请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 手动重新加载模型接口
@app.post("/reload")
async def reload_model():
    """手动触发模型重新加载"""
    try:
        if model_loader.reload_if_needed():
            return {"status": "success", "message": "模型已更新"}
        else:
            return {"status": "success", "message": "模型未发生变化"}
    except Exception as e:
        logger.error(f"模型重新加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型重新加载失败: {str(e)}")

# 主函数，启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
