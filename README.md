# SentinelPay：可解释的实时支付风控平台

> 个人独立开发项目

**项目作者：许甜甜**

**项目性质：个人独立开发、数据科学与机器学习工程实践**

SentinelPay 将一套支付风控模型从「数据生成 → 特征工程 → 时间切分训练 → 阈值优化 → API 服务 → 运营看板」完整串起来。项目不依赖真实交易数据，使用确定性合成数据复现真实业务中的长尾欺诈、时间漂移、设备风险、商户风险和误杀成本。

## 项目亮点

- **业务建模**：同时模拟用户、商户、设备、地理位置、支付渠道和交易时段等风控信号。
- **正确评估**：按时间切分 train / validation / test，避免随机切分带来的未来信息泄漏。
- **不平衡学习**：使用 class weight、PR-AUC、Recall@固定误报率等更贴近风控的指标。
- **业务阈值**：根据“漏放欺诈损失 > 误拦正常交易损失”的成本函数自动选择阈值，而非默认 0.5。
- **可解释性**：返回每笔交易的风险等级、风险因子、建议动作和模型版本。
- **工程化**：FastAPI 在线推理接口、Streamlit 运营看板、Docker Compose、pytest 测试、可复现实验命令。

## 运行方式

### 1. 安装

本项目已在 Windows + Anaconda Python 3.7 环境验证。建议优先使用稳定版 Python 3.11 / 3.12 的新虚拟环境；如果使用本机已经验证过的 Anaconda 环境，直接执行下面的 Anaconda 安装命令即可。

```bash
# 已安装 Anaconda 的 Windows 环境
D:\Anaconda\python.exe -m pip install fastapi==0.85.2 uvicorn==0.18.3 pydantic==1.10.13
```

如果你使用 Python 3.11 / 3.12，请先创建虚拟环境，再执行 `pip install -r requirements.txt`。

### 2. 生成数据并训练

```bash
python -m sentinelpay.pipeline --n-transactions 80000
```

训练后会生成：

- `data/transactions.parquet`：模拟交易明细
- 如果环境没有 `pyarrow`，自动生成 `data/transactions.csv`
- `artifacts/model.joblib`：模型、特征列和业务阈值
- `artifacts/metrics.json`：测试集指标与阈值分析
- `reports/metrics.json`：供看板读取的摘要

### 3. 启动 API

```bash
uvicorn sentinelpay.api:app --reload
```

打开 <http://127.0.0.1:8000/docs> 查看 Swagger。

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/risk/score ^
  -H "Content-Type: application/json" ^
  -d "{\"amount\": 386.5, \"user_id\": \"U00042\", \"merchant_category\": \"electronics\", \"device_id\": \"D1007\", \"country\": \"CN\", \"channel\": \"mobile\", \"hour\": 2, \"account_age_days\": 18, \"is_new_device\": true, \"velocity_1h\": 7}"
```

### 4. 启动看板

```bash
streamlit run dashboard/app.py
```

## 目录结构

```text
sentinelpay/
  config.py       # 路径和随机种子
  data.py         # 可复现交易流生成
  features.py     # 特征工程和 schema 校验
  model.py        # 训练、评估、阈值和解释
  pipeline.py     # 一键运行入口
  api.py          # FastAPI 在线评分
dashboard/
  app.py          # Streamlit 运营看板
tests/
  test_features.py
  test_model.py
```

## 免责声明

本项目只用于学习和技术演示。数据为合成数据，不代表真实金融机构的风险策略。
