# 中文企业年报 RAG 项目

本项目面向中文上市公司年报问答场景，核心链路是：

- 使用 MinerU 解析 PDF 年报
- 将解析结果导入为页级文本和表格块
- 使用 `bge-m3` 构建向量检索
- 使用 `BM25 + 向量检索 + 重排` 进行混合召回
- 使用 Qwen 兼容接口完成最终问答

当前仓库已经从原始实验状态收敛为一条可运行的中文主线，适合继续做：

- 中文年报问答实验
- 交互式提问验证
- 检索、重排、表格问答优化

## 目录结构

```text
RAG_System_Learning/
├── data/
│   └── test_set/
│       ├── pdf_reports/          # 原始 PDF 年报
│       ├── mineru_test/          # MinerU 解析输出
│       ├── subset.csv            # 数据集公司清单
│       └── questions.json        # 评测问题
├── src/
│   ├── api_requests.py
│   ├── embedding_models.py
│   ├── interactive_prompts.py
│   ├── mineru_parsing.py
│   ├── pipeline.py
│   ├── questions_processing.py
│   ├── reranking.py
│   ├── reranking_prompts.py
│   ├── retrieval.py
│   └── text_splitter.py
├── main.py
├── requirements.txt
└── README.md
```

## 环境要求

建议环境：

- Windows
- Python 3.11
- CUDA 可选，显卡环境可显著提升 MinerU 和 embedding 速度

创建虚拟环境并安装依赖：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 环境变量

项目依赖 `.env`。至少需要配置：

```env
QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.6-plus

EMBEDDING_MODEL=D:\AI_Cache\modelscope\BAAI\bge-m3
EMBEDDING_DEVICE=cuda

CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3
CROSS_ENCODER_DEVICE=cuda

HNSW_M=32
HNSW_EF_CONSTRUCTION=80
HNSW_EF_SEARCH=64

TEXT_CHUNK_SIZE=500
TEXT_CHUNK_OVERLAP=80
```

说明：

- `QWEN_BASE_URL` 使用 DashScope 兼容 OpenAI 的接口地址。
- `EMBEDDING_MODEL` 推荐直接配置为本地模型目录，避免重复从 Hugging Face 下载。
- `.env` 不应提交到 Git。

## 快速开始

下面以 `data/test_set` 为例。

### 1. 准备 PDF 年报

将 PDF 放到：

```text
data/test_set/pdf_reports/
```

### 2. 使用 MinerU 解析 PDF

先在项目根目录设置 MinerU 本地模型配置：

```powershell
cd D:\35174\Desktop\RAG_System_Learning
$env:MINERU_TOOLS_CONFIG_JSON=(Resolve-Path '.\magic-pdf.json').Path
$env:MINERU_MODEL_SOURCE='local'
```

单文件解析示例：

```powershell
.\venv\Scripts\mineru.exe -p ".\data\test_set\pdf_reports\TCL智家：2025年年度报告.pdf" -o ".\data\test_set\mineru_test" -m auto -b pipeline -l ch -d cuda
```

如果机器显存较小，不建议一次并发解析多份长年报，建议串行执行。

### 3. 导入 MinerU 结果并建库

注意：下面两个命令需要在数据集目录下执行。

```powershell
cd D:\35174\Desktop\RAG_System_Learning\data\test_set
..\..\venv\Scripts\python.exe ..\..\main.py process-mineru-reports --source-dir .\mineru_test --config qwen_default
```

这一步会完成：

- 导入 MinerU 输出
- 页级文本和表格块切分
- embedding 编码
- 向量库构建
- BM25 索引构建

### 4. 批量跑问题

```powershell
..\..\venv\Scripts\python.exe ..\..\main.py process-questions --config qwen_max_llm_top5
```

结果会输出到当前数据集目录下的 `answers_*.json`。

### 5. 交互式提问

```powershell
..\..\venv\Scripts\python.exe ..\..\main.py interactive-qa --config qwen_max_llm_top5
```

适合手工验证单个问题的效果，例如：

- `TCL智家的董事和高级管理人员都有谁？`
- `TCL智家的公司网址是什么？`
- `TCL智慧家电股份有限公司的使用权资产情况的期末账面价值总计是多少？`

## 当前支持的配置

`process-questions` 和 `interactive-qa` 当前支持以下配置名：

- `qwen_turbo`
- `qwen_max_rerank`
- `qwen_plus_rerank`
- `qwen_max_llm_top5`
- `qwen_max_ce_top5`

一般建议：

- 想先看稳定性：`qwen_max_llm_top5`
- 想测试交叉编码器精排：`qwen_max_ce_top5`

## 当前实现说明

当前系统的主链路是：

1. 公司年报 PDF 解析
2. 页级文本与表格块导入
3. 混合检索
   - BM25
   - 向量检索
   - 重排
4. LLM 最终回答

当前交互链已经支持：

- 公司名归一化匹配
- 多种提问写法的兼容
- 统一回答结构输出

当前仍在继续优化的重点：

- 表格型数值问题的稳定抽取
- 多公司问题的检索与聚合
- 交互问答的证据组织与解释能力

