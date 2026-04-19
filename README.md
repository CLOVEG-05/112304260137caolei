# CL 实验目录

这个目录单独整理的是你之前跑出 `0.96+` 分数的那一版实验，目标提交文件是：

- `submission_tfidf_svm_blend_auc.csv`

这版实验不包含 GitHub 上传结构，只保留纯实验所需的代码、环境说明和结果文件。

## 1. 对应成绩

- Kaggle 提交文件：`submission_tfidf_svm_blend_auc.csv`
- 当时线上成绩：`0.96739`
- 本地验证 AUC（历史记录）：`0.9698208`
- 本目录中也额外保留了我这次用独立脚本重建时生成的文件，便于你后续单独复现实验

## 2. 文件说明

```text
cl/
├─ artifacts/
│  └─ cache/
├─ reports/
│  ├─ tfidf_svm_blend_validation_metrics.json
│  └─ tfidf_svm_blend_validation_metrics_rebuild.json
├─ submissions/
│  ├─ submission_tfidf_svm_blend_auc.csv
│  └─ submission_tfidf_svm_blend_auc_rebuild.csv
├─ README.md
├─ requirements.txt
└─ run_tfidf_svm_blend.py
```

- `run_tfidf_svm_blend.py`：独立实验脚本，直接生成这条 `0.96+` 路线的提交文件
- `requirements.txt`：该实验最小依赖
- `reports/tfidf_svm_blend_validation_metrics.json`：当时保留下来的原始验证记录
- `reports/tfidf_svm_blend_validation_metrics_rebuild.json`：这次在 `cl/` 目录下重建脚本后重新运行得到的验证记录
- `submissions/submission_tfidf_svm_blend_auc.csv`：历史运行产出的原始提交文件
- `submissions/submission_tfidf_svm_blend_auc_rebuild.csv`：这次重建脚本重新生成的提交文件
- `artifacts/cache/`：重跑时自动生成的本地预处理缓存

## 3. 使用的数据

脚本默认读取同级目录下的 Kaggle 原始数据：

```text
../word2vec-nlp-tutorial/
```

也就是这些文件：

- `labeledTrainData.tsv`
- `testData.tsv`

这条路线不需要 `unlabeledTrainData.tsv`，因为它不训练 Word2Vec，而是使用 `TF-IDF + LinearSVC`。

## 4. 环境

当前机器上验证时的 Python 版本是：

```text
Python 3.12.1
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 5. 运行方法

在 `cl/` 目录下运行：

```bash
python run_tfidf_svm_blend.py
```

如果你想强制重建缓存：

```bash
python run_tfidf_svm_blend.py --rebuild-cache
```

如果你想指定别的数据目录：

```bash
python run_tfidf_svm_blend.py --data-dir "E:\machining learning\competition\kaggle1\word2vec-nlp-tutorial"
```

## 6. 这版模型的核心思路

这条路线是一个比较轻量但效果很强的稀疏文本模型：

- 先做文本清洗
- `word TF-IDF + LinearSVC`
- `char TF-IDF + LinearSVC`
- 对两个模型的 `decision_function` 做排名归一化
- 按 `0.8 : 0.2` 融合成最终分数

使用的主要配置为：

- word 模型：`ngram_range=(1,2)`，`C=0.6`
- char 模型：`analyzer=char_wb`，`ngram_range=(3,5)`，`C=0.4`

## 7. 说明

我在原始 `formal` 目录里保留了当时的历史输出文件和验证报告，但没有找到“当时那一刻的独立脚本”原件。  
因此这里的 `run_tfidf_svm_blend.py` 是根据保留下来的配置与结果，重建出的独立实验版脚本。

也就是说：

- `submissions/submission_tfidf_svm_blend_auc.csv` 是原来真实跑出来的历史文件
- `reports/tfidf_svm_blend_validation_metrics.json` 是原来真实保留的验证记录
- `run_tfidf_svm_blend.py` 是为了方便你以后单独复现实验而重建的纯实验脚本

我已经在这个 `cl/` 目录中实际运行过一次重建脚本，当前重建结果为：

- `reports/tfidf_svm_blend_validation_metrics_rebuild.json`
- 本地验证 `ROC-AUC = 0.96676232`

这说明：
- 这份独立脚本可以正常跑通
- 但由于原始独立脚本没有完整保留下来，重建版和历史版并不是逐字节完全一致
- 如果你要保留“当时真正拿去提交 Kaggle 的结果”，请以原始文件 `submission_tfidf_svm_blend_auc.csv` 为准
