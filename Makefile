# KL8 分析工具 Makefile

.PHONY: setup fmt lint test build run train-graph download-data daily walk-forward-kl8 digit-report digit-walk-forward h5-install h5-sync h5-dev h5-build h5-test ci clean help
.DEFAULT_GOAL := help

PYTHON ?= python
RUN ?= uv run --python 3.11 --with-requirements requirements.txt python
REPORT_COUNT ?= 10
GROUP_SIZE ?= 10
OUTPUT_DIR ?= reports
MODE ?= auto
STRATEGY ?=
BATCH_TRIALS ?= 30
WF_PERIODS ?= 300
WF_MIN_TRAIN_SIZE ?= 200
H5_DIR ?= h5
DIGIT_LOTTERY ?= fc3d
DIGIT_CSV ?= data/$(DIGIT_LOTTERY)/data.csv
DIGIT_CANDIDATE_COUNT ?= 10
DIGIT_JSON ?= 1
DIGIT_RANKING_MODE ?= ensemble
DIGIT_ENABLE_MONTE_CARLO ?= 1
DIGIT_MC_SIMULATIONS ?= 20000
DIGIT_ENABLE_ML ?= 1
DIGIT_ML_TRAINING_PERIODS ?= 60
DIGIT_ML_NEGATIVE_SAMPLES ?= 9
DIGIT_CONSTRAINT_MODE ?= soft
DIGIT_CONSTRAINT_PROBABILITY_FLOOR ?= 0.02
DIGIT_CONSTRAINT_PENALTY_WEIGHT ?= 0.05
DIGIT_WF_PERIODS ?= 300
DIGIT_WF_MIN_TRAIN_SIZE ?= 100
DIGIT_WF_OUTPUT_DIR ?= reports/evaluations
DIGIT_WF_BASELINE_RUNS ?= 20
DIGIT_WF_NESTED_TUNING ?= 0
DIGIT_WF_INNER_VALIDATION_PERIODS ?= 10
DIGIT_WF_REPORT_PREFIX ?= digit_walk_forward
DIGIT_WF_ADVANCED_MODELS ?= 1
DIGIT_WF_MC_SIMULATIONS ?= 5000
DIGIT_WF_ML_TRAINING_PERIODS ?= 30
DIGIT_WF_ML_NEGATIVE_SAMPLES ?= 5
DIGIT_WF_COMPARE_WINDOWS ?= 1

setup: ## 安装依赖并准备目录
	@echo "请确认已激活 python311 (conda) 环境"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/kl8', 'results', 'logs']]"
	@echo "依赖安装完成，目录已就绪"

fmt: ## 代码格式化
	@echo "执行 black + isort..."
	$(PYTHON) -m black src tests
	$(PYTHON) -m isort src tests --profile black

lint: ## 静态检查
	@echo "执行 flake8 与 mypy..."
	$(PYTHON) -m flake8 src tests
	$(PYTHON) -m mypy src --ignore-missing-imports

test: ## 运行测试与覆盖率
	@echo "运行 pytest..."
	$(RUN) -m pytest tests -q

build: ## 检查 src/scripts 语法并生成 pyc
	@echo "编译 Python 字节码..."
	$(PYTHON) -m compileall -q src scripts

run: ## 执行示例（需要已有数据文件）
	@echo "运行快乐8高频号码统计示例..."
	$(PYTHON) examples/analysis_example.py

train-graph: ## 训练共现图嵌入缓存
	@echo "训练 Node2Vec 图嵌入..."
	$(PYTHON) scripts/train_graph_embeddings.py --lottery kl8

download-data: ## 下载快乐8历史数据
	$(RUN) scripts/get_data.py --name kl8

daily: download-data ## 更新数据并生成快乐8日报、推荐快照、复盘与累计汇总
	$(RUN) scripts/daily_report.py --count $(REPORT_COUNT) --group-size $(GROUP_SIZE) --output-dir $(OUTPUT_DIR) --mode $(MODE) --batch-trials $(BATCH_TRIALS) $(if $(STRATEGY),--strategy $(STRATEGY),)

walk-forward-kl8: download-data ## 快乐8逐期前推策略回测（无未来数据污染）
	$(RUN) scripts/walk_forward_kl8.py --output-dir $(OUTPUT_DIR) --periods $(WF_PERIODS) --min-train-size $(WF_MIN_TRAIN_SIZE) --count $(REPORT_COUNT) --group-size $(GROUP_SIZE)

digit-report: ## 从本地CSV生成福彩3D/排列三/排列五数字彩报告
	$(RUN) scripts/digit_report.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --candidate-count $(DIGIT_CANDIDATE_COUNT) --ranking-mode $(DIGIT_RANKING_MODE) --monte-carlo-simulations $(DIGIT_MC_SIMULATIONS) --ml-training-periods $(DIGIT_ML_TRAINING_PERIODS) --ml-negative-samples $(DIGIT_ML_NEGATIVE_SAMPLES) --constraint-mode $(DIGIT_CONSTRAINT_MODE) --constraint-probability-floor $(DIGIT_CONSTRAINT_PROBABILITY_FLOOR) --constraint-penalty-weight $(DIGIT_CONSTRAINT_PENALTY_WEIGHT) $(if $(filter 1 true yes,$(DIGIT_JSON)),--json,) $(if $(filter 1 true yes,$(DIGIT_ENABLE_MONTE_CARLO)),,--no-monte-carlo) $(if $(filter 1 true yes,$(DIGIT_ENABLE_ML)),,--no-ml)

digit-walk-forward: ## 数字彩严格逐期前推回测，并对比均匀随机基线
	$(RUN) scripts/digit_walk_forward.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(DIGIT_WF_OUTPUT_DIR) --periods $(DIGIT_WF_PERIODS) --min-train-size $(DIGIT_WF_MIN_TRAIN_SIZE) --candidate-count $(DIGIT_CANDIDATE_COUNT) --baseline-runs $(DIGIT_WF_BASELINE_RUNS) --inner-validation-periods $(DIGIT_WF_INNER_VALIDATION_PERIODS) --report-prefix $(DIGIT_WF_REPORT_PREFIX) --monte-carlo-simulations $(DIGIT_WF_MC_SIMULATIONS) --ml-training-periods $(DIGIT_WF_ML_TRAINING_PERIODS) --ml-negative-samples $(DIGIT_WF_ML_NEGATIVE_SAMPLES) --constraint-mode $(DIGIT_CONSTRAINT_MODE) --constraint-probability-floor $(DIGIT_CONSTRAINT_PROBABILITY_FLOOR) --constraint-penalty-weight $(DIGIT_CONSTRAINT_PENALTY_WEIGHT) $(if $(filter 1 true yes,$(DIGIT_WF_NESTED_TUNING)),--nested-tuning,) $(if $(filter 1 true yes,$(DIGIT_WF_ADVANCED_MODELS)),--advanced-models,) $(if $(filter 1 true yes,$(DIGIT_WF_COMPARE_WINDOWS)),--compare-windows,)

h5-install: ## 安装 Vue3 H5 用户端依赖
	cd $(H5_DIR) && npm install

h5-sync: daily ## 同步最新日报 JSON 到 H5 public/report-data/latest.json
	$(RUN) scripts/sync_h5_data.py --reports-dir $(OUTPUT_DIR) --h5-public-dir $(H5_DIR)/public

h5-dev: h5-sync ## 启动 Vue3 H5 开发服务
	cd $(H5_DIR) && npm run dev

h5-test: ## 运行 Vue3 H5 测试
	cd $(H5_DIR) && npm test

h5-build: h5-sync ## 构建 Vue3 H5 生产产物
	cd $(H5_DIR) && npm run build

ci: fmt lint test build ## 本地 CI
	@echo "本地 CI 全部通过"

clean: ## 清理临时文件和缓存
	@echo "清理缓存与编译文件..."
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(pathlib.Path(name), ignore_errors=True) for name in ['build', 'dist', '.pytest_cache', 'htmlcov']]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

help: ## 查看可用命令
	@echo "可用任务："
	@echo "  make setup          - 安装依赖并初始化目录"
	@echo "  make download-data  - 下载快乐8历史数据"
	@echo "  make daily          - 更新数据并生成日报/推荐快照/复盘/累计汇总"
	@echo "  make walk-forward-kl8 - 快乐8逐期前推策略回测（WF_PERIODS=300）"
	@echo "  make digit-report   - 数字彩日报（默认启用 Monte Carlo/ML，可用 DIGIT_ENABLE_*=0 关闭）"
	@echo "  make digit-walk-forward - 数字彩严格前推（Makefile 默认高级模型；直接 CLI 需 --advanced-models）"
	@echo "  make h5-install     - 安装 Vue3 H5 用户端依赖"
	@echo "  make h5-dev         - 同步数据并启动 Vue3 H5 开发服务"
	@echo "  make h5-build       - 同步数据并构建 Vue3 H5 产物"
	@echo "  make h5-test        - 运行 Vue3 H5 测试"
	@echo "  make fmt            - 格式化代码"
	@echo "  make lint           - 静态检查"
	@echo "  make test           - 运行测试"
	@echo "  make build          - compileall 检查 src 与 scripts"
	@echo "  make run            - 运行示例分析"
	@echo "  make train-graph    - 训练共现图嵌入缓存"
	@echo "  make ci             - 本地 CI（fmt+lint+test+build）"
	@echo "  make clean          - 清理缓存文件"
