# 结构化多因子风险模型构建与风险分析操作手册

## 一、模型设定

- **股票数量**：N
- **因子数量**：K (包括行业因子和CNE6因子）
- **历史时间窗口**：T（日频）
- **市场基准**：沪深300 (csi300)
- **目标**：给定股票持仓，输出如下股票 / 因子的风险指标：
  - 股票的主动风险边际贡献(MCAR)
  - 股票的主动风险贡献(RCAR)
  - 因子的主动风险边际贡献(FMCAR)
  - 因子的主动风险贡献(FRCAR)
- **更新频率**：日频

---

## 二、因子收益率估计（横截面回归）

### 步骤 1：准备因子暴露矩阵 \( X_t \)

在时间窗口中每个时间点 t（t=1,2,...,T），准备所有股票的因子暴露：
- **行业因子暴露**：0/1 变量（31个）
- **风险指数因子暴露**：CNE6因子（38个）

整体逻辑：
  1. 从 qlib 文件系统读取原始字段信息
  2. 计算 CNE6 因子，对因子进行去极值、中性化、正交化、标准化处理
  3. 合并行业因子，得到一个包含行业因子和 CNE6 的因子暴露矩阵

具体执行：
- 第1步。计算 CNE6 因子
    - 1）qlib 原始字段读取及因子计算。参照 cf_quant/data/factor/__init__.py 中的 "CNE6因子计算实例"
    - 2）因子计算并行化。参考 cf_quant/utils/multiprocess.py
    - 3）合并所有因子为一个DataFrame，列为因子名称，行索引为 <instrument, datetime>
    - 4）保存一份到磁盘，parquet 格式
- 第2步。因子预处理
    - 1）去极值。对所有因子按照中位数去极值
    - 2）中性化。对1）中结果进行行业、对数市值中性化。实现逻辑参考 from jqfactor_analyzer import neutralize 函数。删除一个行业以防止多重共线性
    - 3）标准化。对3）中所有因子进行均值为0，标准差为1的标准化
    - 4）去极值和标准化函数引用 cf_quant/utils/preprocess.py 中函数
    - 5）检验因子之间相关性、及因子的VIF(Variance Inflation factor)
    - 6）预处理过程及检验的中间数据都保存一份到磁盘，parquet 格式
- 第3步。合并行业因子
    - 1）行业字段名为 ind_one，取值及对应的名称如下：
    {'801780': '银行', '801180': '房地产', '801230': '综合', '801750': '计算机', '801970': '环保', '801200': '商贸零售', '801890': '机械设备', '801730': '电力设备', '801720': '建筑装饰', '801710': '建筑材料', '801030': '基础化工', '801110': '家用电器', '801130': '纺织服饰', '801010': '农林牧渔', '801080': '电子', '801160': '公用事业', '801150': '医药生物', '801880': '汽车', '801210': '社会服务', '801960': '石油石化', '801050': '有色金属', '801770': '通信', '801170': '交通运输', '801760': '传媒', '801790': '非银金融', '801140': '轻工制造', '801740': '国防军工', '801120': '食品饮料', '801950': '煤炭', '801040': '钢铁', '801980': '美容护理'}
    - 2）将股票行业取值按照 one-hot 形式转换，生成一个DataFrame，列名为行业名，行索引为 <instrument, datetime>
    - 3）将 第2步 得到CNE6因子 与行业因子合并，得到一个包含行业因子和 CNE6因子的因子暴露矩阵 X_t
    - 4）保存一份到磁盘，parquet 格式

注意点：
  1. 将 qlib 的取数逻辑作为一个独立模块，存放在 cf_quant/barra/utils/data_loader.py。确保该模块的可扩展性和简洁性
  2. data_loader.py 字段取数函数中添加时间延长参数。例如：计算时刻 t 因子 f1的值，需要用到5年前的数据，则 qlib.D.features()中的 start_time 需要往前推5年。添加 频率参数，表示 start_time 移动的单位（日/年）

### 步骤 2：横截面回归估计因子收益率 \( b_t \)

在每个时间点 t（t=1,2,...,T）独立进行加权最小二乘回归：

\[
r_t = X_t b_t + u_t
\]

- \( r_t \)：\( N \times 1 \) 股票超额收益率向量
- - \( X_t \)：\( N \times K \) 因子暴露矩阵
- \( b_t \)：\( K \times 1 \) 待估计因子收益率向量
- \( u_t \)：\( N \times 1 \) 特异收益率残差向量

#### 2.1 权重设置

使用**流通市值的平方根**作为回归权重，构建对角权重矩阵 \( W_t \)：

\[
W_t = \text{diag}\left(\sqrt{\text{MV}_{1,t}}, \sqrt{\text{MV}_{2,t}}, \ldots, \sqrt{\text{MV}_{N,t}}\right)
\]

#### 2.2 加权最小二乘估计

\[
\hat{b}_t = (X_t^T W_t X_t)^{-1} X_t^T W_t r_t
\]

#### 2.3 输出

- 因子收益率向量 \( \hat{b}_t \)
- 特异收益率残差 \( \hat{u}_t = r_t - X_t \hat{b}_t \)

---

## 三、因子收益率协方差矩阵 \( F \) 估计

- 将 因子收益率协方差矩阵 F 的估计作为单独的模块（cf_quant/barra/risk_control/covariance.py）
- 多种实现方法可以通过参数进行选择
- 输入：历史各期的因子收益率  \hat{b}_t(t=1,2,...,T) \( K \times 1 \)
- 输出：F：\( K \times K \) 因子协方差矩阵
- 收益率协方差计算前，对\(\hat{b}_t(t=1,2,...,T)\) 按照中位数去极值

### 3.1: 方法1：样本协方差矩阵（基准）

\[
F = \frac{1}{T-1} \sum_{t=1}^T (\hat{b}_t - \bar{b})(\hat{b}_t - \bar{b})^T
\]

其中 \( \bar{b} = \frac{1}{T} \sum_{t=1}^T \hat{b}_t \)

### 3.2: 方法2：Barra 半衰期模型
**核心思想**：方差与相关系数分离估计，半衰期分别平滑

#### 3.2.1: 变量

| 变量                                | 维度 | 说明                                            |
|:----------------------------------| :--- |:----------------------------------------------|
| \(H_C\)                           | 标量 | 相关系数半衰期（交易日），默认 **252**                       |
| \(H_D\)                           | 标量 | 方差半衰期（交易日），默认 **42**                          |
| \(\lambda_C\)                     | 标量 | 相关系数衰减因子. 0.5**(1/H_C)                        |
| \(\lambda_D\)                     | 标量 | 方差衰减因子. 0.5**(1/H_D)                          |
| \(\mathbf{F}_0^{\text{raw}}\)     | \(K \times K\) | 初始协方差矩阵. 前 m (默认20) 期 等权样本协方差 np.cov(\had{b}) |
| \(\sigma_{k,\text{smooth}}^2(0)\) | 标量 | 各因子方差平滑初始值. F_{kk}^{raw}\left( 0 \right)      |

#### 3.2.2 更新原始协方差矩阵 \(\mathbf{F}_T^{\text{raw}}\)（长半衰期）

对每个交易日 \(t = 1, \dots, T\) 执行迭代计算：

\[
\mathbf{F}_t^{\text{raw}} = \lambda_C \cdot \mathbf{F}_{t-1}^{\text{raw}} + (1 - \lambda_C) \cdot (\mathbf{b}_t \mathbf{b}_t^\top)
\]

- \(\mathbf{b}_t \mathbf{b}_t^\top\) 是秩-1矩阵，代表单日协方差贡献。
- \(\mathbf{F}_T^{\text{raw}}\) 是对历史外积的指数加权平均，反映因子间相对稳定的相关结构。


#### 3.2.3 提取相关系数矩阵 \(\mathbf{C}_T\)

由 \(\mathbf{F}_T^{\text{raw}}\) 计算相关系数：

\[
C_{ij}(T) = \frac{F_{ij}^{\text{raw}}(T)}{\sqrt{F_{ii}^{\text{raw}}(T) \cdot F_{jj}^{\text{raw}}(T)}}
\]

- \(\mathbf{C}_T\) 继承长半衰期的稳定性，反映截至 \(T\) 日的因子相关结构。


#### 3.2.4 构建标准差矩阵 \(\mathbf{D}_T\)（短半衰期）

(1) 从 \(\mathbf{F}_t^{\text{raw}}\) 提取各因子方差序列：

\[
V_k(t) = F_{kk}^{\text{raw}}(t), \quad k = 1,\dots,K
\]

(2) 对每个因子 \(k\)，计算平滑方差：

\[
\sigma_{k,\text{smooth}}^2(t) = \lambda_D \cdot \sigma_{k,\text{smooth}}^2(t-1) + (1 - \lambda_D) \cdot V_k(t)
\]

(3) 构造对角矩阵 \(\mathbf{D}_T\)：

\[
\mathbf{D}_T = \text{diag}\left( \sqrt{\sigma_{1,\text{smooth}}^2(T)}, \dots, \sqrt{\sigma_{K,\text{smooth}}^2(T)} \right)
\]

- 二次平滑赋予方差更高的灵敏度（短半衰期），快速响应波动率聚集效应。


#### 3.2.5 合成最终因子协方差矩阵 \(\mathbf{F}_T\)

\[
\mathbf{F}_T = \mathbf{D}_T \cdot \mathbf{C}_T \cdot \mathbf{D}_T
\]

- \(\mathbf{F}_T\) 综合了**长周期相关结构**与**短周期波动水平**，是当前时点的条件协方差预测。

---

#### 3.2.6: 伪代码

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


## 四、特异风险矩阵 \( \Delta \) 估计

### 步骤 4.1：计算历史特异方差

对每期 \( t \)，计算特异方差 \( u_n^2(t) \)。

### 步骤 4.2：分解为 \( S(t) \) 和 \( v_n(t) \)

\[
S(t) = \frac{1}{N} \sum_{n=1}^N u_n^2(t)
\]

\[
v_n(t) = \frac{u_n^2(t)}{S(t)} - 1
\]

满足 \( \frac{1}{N} \sum_{n=1}^N v_n(t) = 0 \)。

### 步骤 4.3：预测 \( S(t+1) \) 使用 ARMA 模型

对 \( S(t) \) 序列建立 ARMA(\( p, q \)) 模型：

\[
S(t) = c + \sum_{i=1}^p \phi_i S(t-i) + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t
\]

得到下一期预测值 \( \hat{S}(t+1) \)。

### 步骤 4.4：预测 \( v_n(t+1) \)

建立面板回归模型：

\[
v_n(t) = \sum_k \beta_{k,n}(t) \cdot \lambda_k(t) + \epsilon_n(t)
\]

- 解释变量：
  - \(\beta_{k,n}(t)\)：行业和CNE6 的因子暴露度，也就是经过数据预处理后的因子暴露矩阵 \( X_t \)
  - \(\lambda_k(t)\)：回归系数
- 估计方法：混合回归（通过cf_quant/utils/preprocess.py脚本中winsorize去极值）。即多期横截面数据混合在一起
- 输出：预测 \( \hat{v}_n(t+1) \)

### 步骤 4.5：合成未来特异方差

\[
\hat{u}_n^2(t+1) = \hat{S}(t+1) \cdot [1 + \hat{v}_n(t+1)]
\]

### 步骤 4.6：构建特异风险矩阵

\[
\Delta_{t+1} = \text{diag}\left( \hat{u}_1^2(t+1), \hat{u}_2^2(t+1), \ldots, \hat{u}_N^2(t+1) \right)
\]

---

## 五、资产协方差矩阵 \( V \)

\[
V_{t+1} = X_{t+1} F_{t+1} X_{t+1}^T + \Delta_{t+1}
\]
 - 其中，F_{t+1}是基于截至 t 日的数据估计的因子协方差矩阵
---

## 六、风险分析（组合层面）

设投资组合权重向量为 \( h_p \)（\( N \times 1 \)），基准权重向量为 \( h_b \)，以沪深300成分股的市值/总市值 作为基准权重向量。

### 6.1 基本定义

- 主动权重：\( h_{PA} = h_p - h_b \)
- 组合因子暴露：\( x_p = X^T h_p \)
- 主动因子暴露：\( x_{PA} = X^T h_{PA} \)

### 6.2 风险度量

- 组合总风险：\( \sigma_p = \sqrt{h_p^T V h_p} \)
- 主动风险：\( \psi_p = \sqrt{h_{PA}^T V h_{PA}} \)

---

## 七、风险边际贡献与风险贡献

### 7.1 股票的主动风险边际贡献（MCAR）

\[
\text{MCAR} = \frac{V h_{PA}}{\psi_p} \quad (N \times 1)
\]

含义：股票 \( n \) 权重增加 1%（从现金融资）导致的主动风险近似变化。

### 7.2 股票的主动风险贡献（RCAR）

\[
\text{RCAR} = h_{PA} \odot \text{MCAR} \quad (N \times 1)
\]

其中 \( \odot \) 表示逐元素相乘。

性质：
\[
\sum_{n=1}^N \text{RCAR}_n = \psi_p
\]

### 7.3 因子的主动风险边际贡献（FMCAR）

\[
\text{FMCAR} = \frac{F x_{PA}}{\psi_p} \quad (K \times 1)
\]

含义：对因子 \( k \) 的主动暴露增加 1 单位导致的主动风险变化（通过因子组合实现）。

### 7.4 因子的主动风险贡献（FRCAR）

\[
\text{FRCAR} = x_{PA} \odot \text{FMCAR} \quad (K \times 1)
\]

性质：
\[
\sum_{k=1}^K \text{FRCAR}_k = \frac{x_{PA}^T F x_{PA}}{\psi_p} = \psi_p - \frac{h_{PA}^T \Delta h_{PA}}{\psi_p}
\]

即主动风险中由因子解释的部分。

---

## 八、风险指标输出到文件

计算完成后，将股票和因子层面的风险分析指标输出为 CSV 文件。

### 8.1 股票风险指标输出

文件格式要求：

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| instrument | str | 股票代码（如：000001.SZ）|
| mcar | float | 主动风险边际贡献（MCAR）|
| rcar | float | 主动风险贡献（RCAR）|
| calc_date | str | 计算日期（格式：YYYY-MM-DD）|

- **每只股票一行**，列数为 4
- **文件命名建议**：`stock_risk_YYYYMMDD.csv`
- **排序**：按 instrument 升序排列
- **数据精度**：float 类型保留 6 位小数
- **编码格式**：UTF-8

### 8.2 因子风险指标输出

文件格式要求：

| 字段名 | 数据类型 | 说明 |
|--------|----------|------|
| factor_name | str | 因子名称（如：LNCAP、BETA）|
| fmcar | float | 因子主动风险边际贡献（FMCAR）|
| frcar | float | 因子主动风险贡献（FRCAR）|
| factor_type | str | 因子类型：行业/规模/波动率/流动性/动量/质量-杠杆/质量-盈利波动/质量-盈利质量/质量-盈利能力/质量-投资质量/价值/成长 |
| calc_date | str | 计算日期（格式：YYYY-MM-DD）|

- **每个因子一行**，列数为 5
- **文件命名建议**：`factor_risk_YYYYMMDD.csv`
- **排序**：先按 factor_type 分组，组内按 factor_name 升序排列
- **数据精度**：float 类型保留 6 位小数
- **编码格式**：UTF-8

### 8.3 输出流程说明

在计算完第六、七章的所有风险指标后，执行以下输出操作：

1. **日频输出**：每日计算完成后，生成当天的股票风险文件和因子风险文件
2. **覆盖策略**：同一日期的文件直接覆盖，不保留历史版本
3. **路径管理**：建议将文件保存至独立的输出目录，便于后续读取和分析


