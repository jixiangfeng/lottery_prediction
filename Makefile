# 福彩3D、排列三、排列五分析工具 Makefile

.PHONY: setup fmt lint test build run digit-fetch digit-report digit-walk-forward digit-probability-walk-forward digit-probability-online ci clean help
.DEFAULT_GOAL := help

-include .env
export

PYTHON ?= python
RUN ?= uv run --python 3.11 --with-requirements requirements-dev.txt python
OUTPUT_DIR ?= reports
DIGIT_LOTTERY ?= fc3d
DIGIT_CSV ?= data/$(DIGIT_LOTTERY)/data.csv
DIGIT_FETCH_PERIODS ?= 1000
DIGIT_CANDIDATE_COUNT ?= 10
DIGIT_JSON ?= 1
DIGIT_FREEZE_PICK ?= 0
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
DIGIT_PROBABILITY_VALIDATION_PERIODS ?= 180
DIGIT_PROBABILITY_MIN_VALIDATION_PERIODS ?= 90
DIGIT_PROBABILITY_MIN_TRAIN_SIZE ?= 100
DIGIT_ONLINE_PERIODS ?= 500
DIGIT_ONLINE_MIN_TRAIN_SIZE ?= 100
DIGIT_ONLINE_TEMPERATURE ?= 0.2
DIGIT_ONLINE_UNIFORM_PRIOR_WEIGHT ?= 0.5
DIGIT_ONLINE_LEARNING_RATE ?= 1.0
DIGIT_ONLINE_FIXED_SHARE ?= 0.01
DIGIT_ONLINE_STATE_PATH ?=
DIGIT_REBUILD_ONLINE_STATE ?= 0

setup: ## 安装依赖并准备运行目录
	@echo "请确认已激活 python311 Conda 环境"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/fc3d', 'data/pl3', 'data/pl5', 'reports', 'logs']]"

fmt: ## 格式化 Python 代码
	$(RUN) -m black src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black

lint: ## 检查格式、导入、代码风格和类型
	$(RUN) -m black --check src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black --check-only
	$(RUN) -m flake8 src tests scripts examples
	$(RUN) -m mypy src

test: ## 运行测试并强制覆盖率不低于 80%
	$(RUN) -m pytest tests -q --cov=src --cov-report=term-missing --cov-report=xml

build: ## 检查 Python 语法并生成字节码
	$(RUN) -m compileall -q src scripts examples

run: ## 运行三种数字彩理论概率示例
	$(RUN) examples/digit_analysis_example.py

digit-fetch: ## 从官方公开接口显式抓取福彩3D或排列三历史
	$(RUN) scripts/fetch_digit_history.py --lottery $(DIGIT_LOTTERY) --periods $(DIGIT_FETCH_PERIODS) --output $(DIGIT_CSV)

digit-report: ## 从本地 CSV 生成数字彩日报
	$(RUN) scripts/digit_report.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --candidate-count $(DIGIT_CANDIDATE_COUNT) --ranking-mode $(DIGIT_RANKING_MODE) --monte-carlo-simulations $(DIGIT_MC_SIMULATIONS) --ml-training-periods $(DIGIT_ML_TRAINING_PERIODS) --ml-negative-samples $(DIGIT_ML_NEGATIVE_SAMPLES) --constraint-mode $(DIGIT_CONSTRAINT_MODE) --constraint-probability-floor $(DIGIT_CONSTRAINT_PROBABILITY_FLOOR) --constraint-penalty-weight $(DIGIT_CONSTRAINT_PENALTY_WEIGHT) --probability-validation-periods $(DIGIT_PROBABILITY_VALIDATION_PERIODS) --probability-min-train-size $(DIGIT_PROBABILITY_MIN_TRAIN_SIZE) --probability-minimum-validation-periods $(DIGIT_PROBABILITY_MIN_VALIDATION_PERIODS) --online-probability-min-train-size $(DIGIT_ONLINE_MIN_TRAIN_SIZE) --online-probability-temperature $(DIGIT_ONLINE_TEMPERATURE) --online-probability-uniform-prior-weight $(DIGIT_ONLINE_UNIFORM_PRIOR_WEIGHT) --online-probability-learning-rate $(DIGIT_ONLINE_LEARNING_RATE) --online-probability-fixed-share $(DIGIT_ONLINE_FIXED_SHARE) $(if $(DIGIT_ONLINE_STATE_PATH),--online-probability-state-path $(DIGIT_ONLINE_STATE_PATH),) $(if $(filter 1 true yes,$(DIGIT_JSON)),--json,) $(if $(filter 1 true yes,$(DIGIT_ENABLE_MONTE_CARLO)),,--no-monte-carlo) $(if $(filter 1 true yes,$(DIGIT_ENABLE_ML)),,--no-ml) $(if $(filter 1 true yes,$(DIGIT_FREEZE_PICK)),--freeze-pick,) $(if $(filter 1 true yes,$(DIGIT_REBUILD_ONLINE_STATE)),--rebuild-online-probability-state,)

digit-walk-forward: ## 严格逐期前推并与随机基线比较
	$(RUN) scripts/digit_walk_forward.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(DIGIT_WF_OUTPUT_DIR) --periods $(DIGIT_WF_PERIODS) --min-train-size $(DIGIT_WF_MIN_TRAIN_SIZE) --candidate-count $(DIGIT_CANDIDATE_COUNT) --baseline-runs $(DIGIT_WF_BASELINE_RUNS) --inner-validation-periods $(DIGIT_WF_INNER_VALIDATION_PERIODS) --report-prefix $(DIGIT_WF_REPORT_PREFIX) --monte-carlo-simulations $(DIGIT_WF_MC_SIMULATIONS) --ml-training-periods $(DIGIT_WF_ML_TRAINING_PERIODS) --ml-negative-samples $(DIGIT_WF_ML_NEGATIVE_SAMPLES) --constraint-mode $(DIGIT_CONSTRAINT_MODE) --constraint-probability-floor $(DIGIT_CONSTRAINT_PROBABILITY_FLOOR) --constraint-penalty-weight $(DIGIT_CONSTRAINT_PENALTY_WEIGHT) $(if $(filter 1 true yes,$(DIGIT_WF_NESTED_TUNING)),--nested-tuning,) $(if $(filter 1 true yes,$(DIGIT_WF_ADVANCED_MODELS)),--advanced-models,) $(if $(filter 1 true yes,$(DIGIT_WF_COMPARE_WINDOWS)),--compare-windows,)

digit-probability-walk-forward: ## 概率 v2 固定校准后的严格前推开发评估
	$(RUN) scripts/digit_probability_walk_forward.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(DIGIT_WF_OUTPUT_DIR) --periods $(DIGIT_WF_PERIODS) --min-train-size $(DIGIT_WF_MIN_TRAIN_SIZE) --candidate-count $(DIGIT_CANDIDATE_COUNT) --validation-periods $(DIGIT_PROBABILITY_VALIDATION_PERIODS) --minimum-validation-periods $(DIGIT_PROBABILITY_MIN_VALIDATION_PERIODS)

digit-probability-online: ## 概率 v3 逐期预测后反馈的500期开发评估
	$(RUN) scripts/digit_probability_online.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(DIGIT_WF_OUTPUT_DIR) --periods $(DIGIT_ONLINE_PERIODS) --min-train-size $(DIGIT_ONLINE_MIN_TRAIN_SIZE) --candidate-count $(DIGIT_CANDIDATE_COUNT) --temperature $(DIGIT_ONLINE_TEMPERATURE) --uniform-prior-weight $(DIGIT_ONLINE_UNIFORM_PRIOR_WEIGHT) --learning-rate $(DIGIT_ONLINE_LEARNING_RATE) --fixed-share $(DIGIT_ONLINE_FIXED_SHARE)

ci: lint test build ## 本地质量闸门
	@echo "本地 CI 全部通过"

clean: ## 清理缓存与构建产物
	$(RUN) -c "import pathlib, shutil; [shutil.rmtree(path, ignore_errors=True) for name in ['build', 'dist', '.pytest_cache', 'htmlcov'] for path in [pathlib.Path(name)]]; [shutil.rmtree(path, ignore_errors=True) for path in pathlib.Path('.').rglob('__pycache__')]"

help: ## 查看可用命令
	@echo "make setup                安装依赖"
	@echo "make run                  运行最小示例"
	@echo "make digit-fetch          抓取福彩3D或排列三官方历史"
	@echo "make digit-report         生成数字彩日报"
	@echo "make digit-walk-forward   执行严格前推回测"
	@echo "make digit-probability-walk-forward 执行概率 v2 开发评估"
	@echo "make digit-probability-online 执行概率 v3 在线反馈评估"
	@echo "make ci                   运行完整质量闸门"
