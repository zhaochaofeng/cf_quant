# 结构化多因子风险模型构建与风险分析

## 一、模型设定

- **股票数量**：N
- **因子数量**：K (包括行业因子和CNE6因子）
- **历史时间窗口**：T（日频，交易日）
- **市场基准**：沪深300 (csi300)
- **目标**：给定股票持仓，输出风险指标：
  - 所有股票的主动风险边际贡献(MCAR)
  - 所有股票的主动风险贡献(RCAR) (没有持仓的股票值为0)
  - 因子的主动风险边际贡献(FMCAR)
  - 因子的主动风险贡献(FRCAR)
- **股票持仓**： 
  - 'random': 随机从市场基准中选择50只股票，持仓100股（用于测试）
  - dict[str, int]: 传入股票代码及持仓数量
- **更新频率**：日频

---

## 二、因子收益率估计

### 步骤 1：因子暴露矩阵 \( X \)

准备每个时间点 t（t=1,2,...,T）的股票因子暴露：
- 行业因子暴露：0/1 变量（31个）
- 风格因子暴露：CNE6因子（38个）

整体逻辑：
1. 从 qlib 文件系统读取原始字段
2. 计算 CNE6 因子，对因子进行去极值、行业/市值中性化、标准化处理
3. 合并行业因子，得到一个包含行业因子和 CNE6 因子的暴露矩阵

具体执行：
1. 计算 CNE6 因子
  - 1）qlib 原始字段读取及因子计算。参照 cf_quant/data/factor/__init__.py 中的 "CNE6因子计算实例"
  - 2）因子计算并行化。参考 cf_quant/utils/multiprocess.py
  - 3）合并所有因子为一个DataFrame，列为因子名称，行索引为 <instrument, datetime>
  - 4）保存一份到磁盘，parquet 格式
2. 因子预处理 
  - 1）去极值。对所有因子按照中位数去极值
  - 2）中性化。对1）中结果进行行业、对数市值中性化。实现逻辑参考 from jqfactor_analyzer import neutralize 函数。删除一个行业以防止多重共线性
  - 3）标准化。对3）中所有因子进行均值为0，标准差为1的标准化
  - 4）去极值和标准化函数引用 cf_quant/utils/preprocess.py 中函数
  - 5）检验因子之间相关性、及因子的VIF(Variance Inflation factor)
  - 6）预处理过程及检验的中间数据都保存一份到磁盘，parquet 格式
3. 合并行业因子
  - 1）行业字段名为 ind_one，取值及对应的名称如下：
  {'801780': '银行', '801180': '房地产', '801230': '综合', '801750': '计算机', '801970': '环保', '801200': '商贸零售', '801890': '机械设备', '801730': '电力设备', '801720': '建筑装饰', '801710': '建筑材料', '801030': '基础化工', '801110': '家用电器', '801130': '纺织服饰', '801010': '农林牧渔', '801080': '电子', '801160': '公用事业', '801150': '医药生物', '801880': '汽车', '801210': '社会服务', '801960': '石油石化', '801050': '有色金属', '801770': '通信', '801170': '交通运输', '801760': '传媒', '801790': '非银金融', '801140': '轻工制造', '801740': '国防军工', '801120': '食品饮料', '801950': '煤炭', '801040': '钢铁', '801980': '美容护理'}
  - 2）将股票行业取值按照 one-hot 形式转换，生成一个DataFrame，列名为行业名，行索引为 <instrument, datetime>
  - 3）将 与得到CNE6因子合并，得到一个包含行业因子和 CNE6因子的因子暴露矩阵 X
  - 4）保存一份到磁盘，parquet 格式

注意点：
  1. 将 qlib 的取数逻辑作为一个独立模块，存放在 cf_quant/barra/utils/data_loader.py。确保该模块的可扩展性和简洁性
  2. data_loader.py 字段取数函数中添加时间延长参数。如：计算时刻 t 因子 f1 的值，需要用到5年前的数据，则 qlib.D.features()中的 start_time 需要往前移动 5 年。添加频率参数，表示 start_time 移动的单位（日/年）

### 步骤 2：估计因子收益率 \( b_t \)

支持两种回归方法，通过 `fit_multi_periods(method=...)` 参数选择，默认为 `constrained`。

---
#### 方法一：WLS（不带约束）

在每个时间点 t（t=1,2,...,T）独立进行加权最小二乘回归：

\[
r_t = X_t b_t + u_t
\]

- \( r_t \)：\( N \times 1 \) 股票超额收益率向量
- \( X_t \)：\( N \times K \) 因子暴露矩阵（含全部行业因子 + CNE6 风格因子）
- \( b_t \)：\( K \times 1 \) 待估计因子收益率向量
- \( u_t \)：\( N \times 1 \) 特异收益率向量

**权重设置**：使用**流通市值的平方根**作为回归权重：

\[
W_t = \text{diag}\left(\sqrt{\text{MV}_{1,t}}, \sqrt{\text{MV}_{2,t}}, \ldots, \sqrt{\text{MV}_{N,t}}\right)
\]

**估计**：

\[
\hat{b}_t = (X_t^T W_t X_t)^{-1} X_t^T W_t r_t
\]

**特点**：不加截距项，全部行业因子保留。行业因子收益率解释为该行业的绝对超额收益。

---
#### 方法二：Constrained WLS（带行业流通市值加权和为 0 约束）

模型设定同方法一，但**添加截距项**，同时对行业因子施加线性约束：

\[
r_t = c_t + X_t b_t + u_t
\]

\[
\text{s.t. } \sum_{j} \frac{\text{MV}_{j,t}^{\text{ind}}}{\sum_i \text{MV}_{i,t}^{\text{ind}}} \cdot b_{j,t} = 0
\]

- \( c_t \)：截距项，代表市值加权市场基准收益
- 约束条件中权重为各行业的**流通市值之和**（非平方根），与回归权重区分
- 截距项不纳入输出（仅用于因子风险分析，不需要截距因子）

**特点**：行业因子收益率含义为"相对市值加权基准的偏离"，整体加权和为 0，可解释性更强。

---
#### 输出（两种方法一致）

- 因子收益率向量 \( \hat{b}_t \)（K × 1，含行业因子 + CNE6 风格因子，不含截距）
- 特异收益率向量 \( \hat{u}_t = r_t - \hat{r}_t \)（N × 1）
- 保存到磁盘，parquet格式

---

## 三、因子收益率协方差矩阵 \( F \) 估计

- 将 因子收益率协方差矩阵 F 的估计作为单独的模块（cf_quant/barra/risk_control/covariance.py）
- 多种实现方法可以选择
- 输入：因子收益率向量  \hat{b}_t (t=1,2,...,T) \( K \times 1 \)
- 输出：F \( K \times K \) 因子协方差矩阵
- 计算前，对\(\hat{b}_t(t=1,2,...,T)\) 以时间维度按照中位数去极值

### 方法1：样本协方差矩阵（基准）

\[
F = \frac{1}{T-1} \sum_{t=1}^T (\hat{b}_t - \bar{b})(\hat{b}_t - \bar{b})^T
\]

其中 \( \bar{b} = \frac{1}{T} \sum_{t=1}^T \hat{b}_t \)

### 方法2：Barra 半衰期模型
**核心思想**：方差与相关系数分离估计，半衰期分别平滑

#### 变量

| 变量                                | 维度 | 说明                                                  |
|:----------------------------------| :--- |:----------------------------------------------------|
| \(H_C\)                           | 标量 | 相关系数半衰期（交易日），默认 **252**                             |
| \(H_D\)                           | 标量 | 方差半衰期（交易日），默认 **42**                                |
| \(\lambda_C\)                     | 标量 | 相关系数衰减因子. 0.5**(1/H_C)                              |
| \(\lambda_D\)                     | 标量 | 方差衰减因子. 0.5**(1/H_D)                                |
| \(\mathbf{F}_0^{\text{raw}}\)     | \(K \times K\) | 初始协方差矩阵. 前 m (默认为因子个数的3倍) 期 等权样本协方差 np.cov(\had{b}) |
| \(\sigma_{k,\text{smooth}}^2(0)\) | 标量 | 各因子方差平滑初始值. F_{kk}^{raw}\left( 0 \right)            |

---
#### 更新原始协方差矩阵 \(\mathbf{F}_T^{\text{raw}}\)（长半衰期）

对每个交易日 \(t = 1, \dots, T\) 执行迭代计算：

\[
\mathbf{F}_t^{\text{raw}} = \lambda_C \cdot \mathbf{F}_{t-1}^{\text{raw}} + (1 - \lambda_C) \cdot (\mathbf{b}_t \mathbf{b}_t^\top)
\]

- \(\mathbf{b}_t \mathbf{b}_t^\top\) 是秩-1矩阵，代表单日协方差贡献。
- \(\mathbf{F}_T^{\text{raw}}\) 是对历史外积的指数加权平均，反映因子间相对稳定的相关结构。

---
#### 提取相关系数矩阵 \(\mathbf{C}_T\)

由 \(\mathbf{F}_T^{\text{raw}}\) 计算相关系数：

\[
C_{ij}(T) = \frac{F_{ij}^{\text{raw}}(T)}{\sqrt{F_{ii}^{\text{raw}}(T) \cdot F_{jj}^{\text{raw}}(T)}}
\]

- \(\mathbf{C}_T\) 继承长半衰期的稳定性，反映截至 \(T\) 日的因子相关结构
- 对角线元素为1（不包含方差信息）

---

#### 构建标准差矩阵 \(\mathbf{D}_T\)（短半衰期）

1. 从 \(\mathbf{F}_t^{\text{raw}}\) 提取各因子方差序列：

\[
V_k(t) = F_{kk}^{\text{raw}}(t), \quad k = 1,\dots,K
\]

2. 对每个因子 \(k\)，计算平滑方差：

\[
\sigma_{k,\text{smooth}}^2(t) = \lambda_D \cdot \sigma_{k,\text{smooth}}^2(t-1) + (1 - \lambda_D) \cdot V_k(t)
\]

3. 构造对角矩阵 \(\mathbf{D}_T\)：

\[
\mathbf{D}_T = \text{diag}\left( \sqrt{\sigma_{1,\text{smooth}}^2(T)}, \dots, \sqrt{\sigma_{K,\text{smooth}}^2(T)} \right)
\]

- 二次平滑赋予方差更高的灵敏度（短半衰期），快速响应波动率聚集效应。

---

#### 合成最终因子协方差矩阵 \(\mathbf{F}\)

\[
\mathbf{F} = \mathbf{D}_T \cdot \mathbf{C}_T \cdot \mathbf{D}_T
\]

- \(\mathbf{F}\) 综合了**长周期相关结构**与**短周期波动水平**，是当前时点的条件协方差预测。
- 对角线元素：\sigma_{jj}^{2} , j=1,2,...,K. 表示因子标准差信息
- 非对角线元素：\sigma_{i} \sigma_{j} c_{ij}。i,j=1,2,...,K. 表示因子协方差信息
---

#### 伪代码

```python
import numpy as np

def ewma_update(old_matrix, new_outer, lam):
    return lam * old_matrix + (1 - lam) * new_outer

# 初始化
F_raw = initial_covariance          # K×K 初始协方差矩阵
var_smooth = np.diag(F_raw).copy()  # 各因子平滑方差初值

# 每日更新（t = 1..T）
for b_t in daily_factor_returns:
    outer = np.outer(b_t, b_t)
    
    # 1. 更新原始协方差矩阵
    F_raw = ewma_update(F_raw, outer, lambda_C)
    
    # 2. 提取基础方差并二次平滑
    V_t = np.diag(F_raw)
    var_smooth = lambda_D * var_smooth + (1 - lambda_D) * V_t
    
# 最终输出
C = F_raw / np.sqrt(np.outer(np.diag(F_raw), np.diag(F_raw)))
D = np.diag(np.sqrt(var_smooth))
F_final = D @ C @ D
```

---

## 四、特异收益率协方差矩阵 \( \Delta \) 估计

### 步骤 1：计算历史特异方差

对每期 \( t \)(t=1,2,...,T)，计算特异方差 \( u_n^2(t) \). 通过 特异收益率 \hat{u}_t 进行计算

### 步骤 2：分解为 \( S(t) \) 和 \( v_n(t) \)

\[
S(t) = \frac{1}{N} \sum_{n=1}^N u_n^2(t)
\]

\[
v_n(t) = \frac{u_n^2(t)}{S(t)} - 1
\]

满足 \( \frac{1}{N} \sum_{n=1}^N v_n(t) = 0 \)

### 步骤 3：预测 \( S(T+1) \) .使用 ARMA 模型

对 \( S(t) \) 序列建立 ARMA(\( p, q \)) 模型（默认 p=1,q=1）：

\[
S(t) = c + \sum_{i=1}^p \phi_i S(t-i) + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t
\]

- 训练：使用 t \in (1, T) 区间数据
- 预测： \( \hat{S}(T+1) \)

### 步骤 4：预测 \( v_n(T+1) \)

建立面板回归模型：以因子暴露 X 作为自变量，v_{n} 作为因变量：

\[
\ v_{n}=\sum_{k} X_{k,n}\cdot \lambda_{k} +\epsilon_{n}
\]

- 解释变量：
  - \(X_{k,n}\)：行业和CNE6 的因子暴露度
  - \(\lambda_k(t)\)：回归系数
- 估计方法：混合回归，即多期（默认 120个交易日）横截面数据混合在一起。回归之前对 v_{n} 进行中位数去极值处理
- 训练：使用 t \in (1, T-1) 区间数据
- 输出：预测 \( \hat{v}_n(T) \)。由于因子暴露在短期内变化很小，所以将 \( \hat{v}_n(T) \) 作为 \( \hat{v}_n(T+1) \) 的估计

### 步骤 5：合成未来特异方差

\[
\hat{u}_n^2(T+1) = \hat{S}(T+1) \cdot [1 + \hat{v}_n(T+1)]
\]

### 步骤 6：构建特异风险矩阵

\[
\Delta = \text{diag}\left( \hat{u}_1^2(T+1), \hat{u}_2^2(T+1), \ldots, \hat{u}_N^2(T+1) \right)
\]

---

## 五、资产协方差矩阵 \( V \)

\[
V = X_{T} F X_{T}^T + \Delta
\]

---

## 六、风险分析（组合层面）

设投资组合权重向量为 \( h_p \)（\( N \times 1 \)），基准权重向量为 \( h_b \)，以市场基准指数的成分股的市值/总市值 作为基准权重向量。

### 6.1 基本定义

- 主动权重：\( h_{PA} = h_p - h_b \) 
- 组合因子暴露：\( x_p = X^T h_p \)
- 主动因子暴露：\( x_{PA} = X^T h_{PA} \)
- 主动风险：\( \psi_p = \sqrt{h_{PA}^T V h_{PA}} \)

### 6.2 股票的主动风险边际贡献（MCAR）

\[
\text{MCAR} = \frac{V h_{PA}}{\psi_p} \quad (N \times 1)
\]

含义：股票 \( n \) 权重增加 1% 引起的主动风险的变化

### 6.3 股票的主动风险贡献（RCAR）

\[
\text{RCAR} = h_{PA} \odot \text{MCAR} \quad (N \times 1)
\]

其中 \( \odot \) 表示逐元素相乘。

性质：
\[
\sum_{n=1}^N \text{RCAR}_n = \psi_p
\]

### 6.4 因子的主动风险边际贡献（FMCAR）

\[
\text{FMCAR} = \frac{F x_{PA}}{\psi_p} \quad (K \times 1)
\]

含义：对因子 \( k \) 的主动暴露增加 1 单位导致的主动风险变化。

### 6.5 因子的主动风险贡献（FRCAR）

\[
\text{FRCAR} = x_{PA} \odot \text{FMCAR} \quad (K \times 1)
\]


---

## 七、风险指标输出
计算完成后，将股票和因子层面的风险分析指标输出为 CSV 文件并写入mysql 表，字段信息见 risk_control.sql

