# 福彩3D、排列三 learned ranker Makefile

.PHONY: setup fmt lint test build run digit-fetch digit-reconcile-jsonl digit-predict-today three-layer-acceptance digit-predictability-audit digit-online-gradient digit-behavioral-context digit-learned-ranker-train digit-learned-ranker-adaptive digit-learned-ranker-evaluate digit-learned-ranker-daily ci clean help
.DEFAULT_GOAL := help

-include .env
export

PYTHON ?= python3
RUN ?= uv run --python 3.11 --with-requirements requirements-dev.txt python
OUTPUT_DIR ?= reports
DIGIT_LOTTERY ?= fc3d
DIGIT_CSV ?= data/$(DIGIT_LOTTERY)/official_history.csv
DIGIT_FETCH_PERIODS ?= 0
DIGIT_RAW_JSONL ?= data/$(DIGIT_LOTTERY)/raw/history.jsonl
THREE_LAYER_ACCEPTANCE ?= $(OUTPUT_DIR)/acceptance/three_layer_$(DIGIT_LOTTERY).json
DIGIT_V4_PARAMS ?= $(OUTPUT_DIR)/state/learned_ranker_v4/$(DIGIT_LOTTERY)_params.json
DIGIT_V4_EVALUATION ?= $(OUTPUT_DIR)/evaluations/learned_ranker_v4_$(DIGIT_LOTTERY).json
DIGIT_V4_MIN_TRAIN_SIZE ?= 150
DIGIT_V4_RANDOM_TRIALS ?= 24
DIGIT_V4_LOCAL_TRIALS ?= 12
DIGIT_V4_EVALUATION_STRIDE ?= 1
DIGIT_V4_FROZEN_TEST_PERIODS ?= 500
DIGIT_V4_OBJECTIVE_PROFILE ?= research_calibrated
DIGIT_V4_ADAPTIVE_OUTER_PERIODS ?= 500
DIGIT_V4_ADAPTIVE_RETRAIN_INTERVAL ?= 10
DIGIT_V4_ADAPTIVE_TRAINING_LOOKBACK ?= 500
DIGIT_V4_SMOKE ?= 0
DIGIT_BEHAVIORAL_OUTPUT ?= $(OUTPUT_DIR)/development/behavioral_context_v2_$(DIGIT_LOTTERY).json

setup: ## 准备运行目录并验证 Python 3.11
	$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/fc3d', 'data/pl3', 'reports', 'logs']]"

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

run: ## 运行理论概率示例
	$(RUN) examples/digit_analysis_example.py

digit-fetch: ## 从固定白名单抓取并追加原始JSONL，不直接覆盖CSV
	$(RUN) scripts/fetch_digit_history.py --lottery $(DIGIT_LOTTERY) --periods $(DIGIT_FETCH_PERIODS) --output-jsonl $(DIGIT_RAW_JSONL)

digit-reconcile-jsonl: ## 多源原始JSONL对账后生成标准CSV
	$(RUN) scripts/reconcile_digit_jsonl.py --raw-jsonl $(DIGIT_RAW_JSONL) --output-csv $(DIGIT_CSV)

digit-predict-today: ## 抓取最新开奖并基于锁定影子状态输出下一期研究Top50
	$(RUN) scripts/digit_predict_today.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV)

three-layer-acceptance: ## 不训练v4、不读Frozen结果的三层离线验收
	$(RUN) scripts/three_layer_acceptance.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --output $(THREE_LAYER_ACCEPTANCE)

digit-learned-ranker-train: ## Search/Validation 参数搜索并锁定参数
	$(RUN) scripts/digit_learned_ranker.py train --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --params $(DIGIT_V4_PARAMS) --min-train-size $(DIGIT_V4_MIN_TRAIN_SIZE) --random-trials $(DIGIT_V4_RANDOM_TRIALS) --local-trials $(DIGIT_V4_LOCAL_TRIALS) --evaluation-stride $(DIGIT_V4_EVALUATION_STRIDE) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --objective-profile $(DIGIT_V4_OBJECTIVE_PROFILE) $(if $(filter 1 true yes,$(DIGIT_V4_SMOKE)),--smoke,)

digit-learned-ranker-adaptive: ## 开发区逐期预测，每10期重选参数，无信号则放弃
	$(RUN) scripts/digit_learned_ranker.py adaptive --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --outer-periods $(DIGIT_V4_ADAPTIVE_OUTER_PERIODS) --retrain-interval $(DIGIT_V4_ADAPTIVE_RETRAIN_INTERVAL) --training-lookback $(DIGIT_V4_ADAPTIVE_TRAINING_LOOKBACK) $(if $(filter 1 true yes,$(DIGIT_V4_SMOKE)),--smoke,)

digit-predictability-audit: ## 开发区时序置换和简单基线可预测性审计
	$(RUN) scripts/digit_predictability_audit.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --output $(OUTPUT_DIR)/development/predictability_audit_$(DIGIT_LOTTERY).json

digit-online-gradient: ## v4逐期特征归因和正则化在线梯度开发对照
	$(RUN) scripts/digit_online_gradient.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --output $(OUTPUT_DIR)/development/online_gradient_v4_$(DIGIT_LOTTERY).json

digit-behavioral-context: ## 在Frozen以前按日常Top50口径运行v4行为A/B/C
	$(RUN) scripts/digit_behavioral_context.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --all-development-blocks --output $(DIGIT_BEHAVIORAL_OUTPUT)

digit-learned-ranker-evaluate: ## 使用锁定参数一次性评估 Frozen Test
	$(RUN) scripts/digit_learned_ranker.py evaluate --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --params $(DIGIT_V4_PARAMS)

digit-learned-ranker-daily: ## 生成研究日报和不可覆盖快照
	$(RUN) scripts/digit_learned_ranker.py daily --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --params $(DIGIT_V4_PARAMS) --evaluation $(DIGIT_V4_EVALUATION)

ci: lint test build ## 本地质量闸门
	@echo "本地 CI 全部通过"

clean: ## 清理缓存与构建产物
	$(RUN) -c "import pathlib, shutil; [shutil.rmtree(path, ignore_errors=True) for name in ['build', 'dist', '.pytest_cache', 'htmlcov'] for path in [pathlib.Path(name)]]; [shutil.rmtree(path, ignore_errors=True) for path in pathlib.Path('.').rglob('__pycache__')]"

help: ## 查看可用命令
	@echo "make digit-fetch                    抓取官方历史"
	@echo "make digit-reconcile-jsonl          多源JSONL对账生成CSV"
	@echo "make digit-predict-today            抓最新开奖并输出下一期研究Top50"
	@echo "make three-layer-acceptance         运行不读取Frozen结果的三层验收"
	@echo "make digit-learned-ranker-train     Search/Validation 调参"
	@echo "make digit-learned-ranker-adaptive  在线自适应开发模拟"
	@echo "make digit-predictability-audit     开发区可预测性审计"
	@echo "make digit-online-gradient          v4逐期归因和在线梯度对照"
	@echo "make digit-behavioral-context       v4行为特征日常口径A/B/C"
	@echo "make digit-learned-ranker-evaluate  Frozen Test 评估"
	@echo "make digit-learned-ranker-daily     生成研究日报"
	@echo "make ci                             运行完整质量闸门"
