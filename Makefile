# 福彩3D、排列三 learned ranker Makefile

.PHONY: setup fmt lint test build run digit-fetch digit-reconcile-jsonl digit-predict-today three-layer-acceptance digit-predictability-audit digit-online-gradient digit-probability-v5-register digit-probability-v5-development digit-probability-v5-null-smoke digit-model-scoreboard digit-learned-ranker-train digit-learned-ranker-adaptive digit-learned-ranker-evaluate digit-learned-ranker-daily kl8-fetch kl8-fetch-csv kl8-pick4-predict-today kl8-pick4-test-today kl8-pick4-rank-challenger kl8-pick5-register kl8-pick5-development kl8-pick5-null-smoke kl8-pick5-null-formal kl8-pick5-predict-today kl8-feature-discovery-v2 ci clean help
.DEFAULT_GOAL := help

-include .env
export

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
DIGIT_V4_DIRECT_OBJECTIVE_TOP_K ?= 50
DIGIT_V4_ADAPTIVE_OUTER_PERIODS ?= 500
DIGIT_V4_ADAPTIVE_RETRAIN_INTERVAL ?= 10
DIGIT_V4_ADAPTIVE_TRAINING_LOOKBACK ?= 500
DIGIT_V4_SMOKE ?= 0
DIGIT_SCOREBOARD_OUTPUT ?= $(OUTPUT_DIR)/development/model_scoreboard_20260721.json
DIGIT_V5_DEVELOPMENT_OUTPUT ?= $(OUTPUT_DIR)/development/probability_v5_$(DIGIT_LOTTERY).json
DIGIT_V5_PROTOCOL ?= $(OUTPUT_DIR)/development/probability_v5_protocol_$(DIGIT_LOTTERY).json
DIGIT_V5_NULL_OUTPUT ?= $(OUTPUT_DIR)/development/probability_v5_null_smoke_$(DIGIT_LOTTERY).json
DIGIT_V5_NULL_SMOKE_ITERATIONS ?= 2
DIGIT_V5_NULL_WORKERS ?= 1
DIGIT_V5_FROZEN_TEST_PERIODS ?= 500
DIGIT_V5_SMOKE ?= 0
KL8_CSV ?= data/kl8/kl8.csv
KL8_RAW_JSONL ?= data/kl8/raw/history.jsonl
KL8_FETCH_PERIODS ?= 0
KL8_FROZEN_PERIODS ?= 500
KL8_PROTOCOL ?= $(OUTPUT_DIR)/development/kl8_pick5_protocol_v1.json
KL8_DEVELOPMENT_OUTPUT ?= $(OUTPUT_DIR)/development/kl8_pick5_development_v1.json
KL8_NULL_OUTPUT ?= $(OUTPUT_DIR)/development/kl8_pick5_null_smoke_v1.json
KL8_NULL_CHECKPOINT ?= $(OUTPUT_DIR)/development/kl8_pick5_null_checkpoints
KL8_NULL_SMOKE_ITERATIONS ?= 2
KL8_NULL_WORKERS ?= 8
KL8_FORMAL_NULL_OUTPUT ?= $(OUTPUT_DIR)/development/kl8_pick5_null_formal_v1.json
KL8_FORMAL_NULL_CHECKPOINT ?= $(OUTPUT_DIR)/development/kl8_pick5_null_formal_checkpoints
KL8_FORMAL_NULL_ITERATIONS ?= 5000
KL8_SMOKE ?= 0
KL8_FEATURE_DISCOVERY_V2_OUTPUT ?= $(OUTPUT_DIR)/development/kl8_feature_discovery_v2.json
KL8_FEATURE_DISCOVERY_V2_N_JOBS ?= 1
KL8_PICK4_TEST_TICKETS ?= 5
KL8_PICK4_RANK_OUTPUT ?= $(OUTPUT_DIR)/development/kl8_pick4_rank_challenger_v2.json
KL8_PICK4_RANK_N_JOBS ?= 1

setup: ## 准备运行目录并验证 Python 3.11
	$(RUN) -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/fc3d', 'data/pl3', 'reports', 'logs']]"

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
	$(RUN) scripts/digit_learned_ranker.py train --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --params $(DIGIT_V4_PARAMS) --min-train-size $(DIGIT_V4_MIN_TRAIN_SIZE) --random-trials $(DIGIT_V4_RANDOM_TRIALS) --local-trials $(DIGIT_V4_LOCAL_TRIALS) --evaluation-stride $(DIGIT_V4_EVALUATION_STRIDE) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --objective-profile $(DIGIT_V4_OBJECTIVE_PROFILE) --direct-objective-top-k $(DIGIT_V4_DIRECT_OBJECTIVE_TOP_K) $(if $(filter 1 true yes,$(DIGIT_V4_SMOKE)),--smoke,)

digit-learned-ranker-adaptive: ## 开发区逐期预测，每10期重选参数，无信号则放弃
	$(RUN) scripts/digit_learned_ranker.py adaptive --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --output-dir $(OUTPUT_DIR) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --outer-periods $(DIGIT_V4_ADAPTIVE_OUTER_PERIODS) --retrain-interval $(DIGIT_V4_ADAPTIVE_RETRAIN_INTERVAL) --training-lookback $(DIGIT_V4_ADAPTIVE_TRAINING_LOOKBACK) $(if $(filter 1 true yes,$(DIGIT_V4_SMOKE)),--smoke,)

digit-predictability-audit: ## 开发区时序置换和简单基线可预测性审计
	$(RUN) scripts/digit_predictability_audit.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --output $(OUTPUT_DIR)/development/predictability_audit_$(DIGIT_LOTTERY).json

digit-online-gradient: ## v4逐期特征归因和正则化在线梯度开发对照
	$(RUN) scripts/digit_online_gradient.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V4_FROZEN_TEST_PERIODS) --output $(OUTPUT_DIR)/development/online_gradient_v4_$(DIGIT_LOTTERY).json

digit-probability-v5-register: ## 只写一次登记v5完整开发协议
	$(RUN) scripts/digit_probability_v5.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V5_FROZEN_TEST_PERIODS) --protocol $(DIGIT_V5_PROTOCOL) --register-protocol

digit-probability-v5-development: ## v5隔离开发挑战器，不读取Frozen或模型状态
	$(RUN) scripts/digit_probability_v5.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V5_FROZEN_TEST_PERIODS) --output $(DIGIT_V5_DEVELOPMENT_OUTPUT) $(if $(filter 1 true yes,$(DIGIT_V5_SMOKE)),--smoke,--protocol $(DIGIT_V5_PROTOCOL))

digit-probability-v5-null-smoke: ## 少量完整重放随机模拟，只验证执行链
	$(RUN) scripts/digit_probability_v5_null.py --lottery $(DIGIT_LOTTERY) --csv $(DIGIT_CSV) --frozen-test-periods $(DIGIT_V5_FROZEN_TEST_PERIODS) --output $(DIGIT_V5_NULL_OUTPUT) --iterations $(DIGIT_V5_NULL_SMOKE_ITERATIONS) --workers $(DIGIT_V5_NULL_WORKERS) --smoke

kl8-fetch: ## 从固定福彩官网接口追加快乐8原始JSONL
	$(RUN) scripts/kl8_fetch_history.py --periods $(KL8_FETCH_PERIODS) --output-jsonl $(KL8_RAW_JSONL)

kl8-fetch-csv: ## 显式创建快乐8标准CSV；已有不同内容时拒绝覆盖
	$(RUN) scripts/kl8_fetch_history.py --periods $(KL8_FETCH_PERIODS) --output-csv $(KL8_CSV)

kl8-pick4-predict-today: ## 快乐8选4安全边界；无准入时正式候选为空
	$(RUN) scripts/kl8_pick4_predict_today.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS)

kl8-pick4-test-today: ## 快乐8选4等概率娱乐测试组合，不是正式推荐
	$(RUN) scripts/kl8_pick4_predict_today.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS) --test --ticket-count $(KL8_PICK4_TEST_TICKETS)

kl8-pick4-rank-challenger: ## 一次性Pick4 LambdaRank@4开发挑战；失败即关闭
	$(RUN) scripts/kl8_pick4_rank_challenger.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS) --output $(KL8_PICK4_RANK_OUTPUT) --n-jobs $(KL8_PICK4_RANK_N_JOBS)

kl8-pick5-register: ## 只写一次登记快乐8选5完整开发协议
	$(RUN) scripts/kl8_pick5_development.py --csv $(KL8_CSV) --protocol $(KL8_PROTOCOL) --frozen-periods $(KL8_FROZEN_PERIODS) --register-protocol

kl8-pick5-development: ## 快乐8选5开发挑战器；smoke不声明已登记
	$(RUN) scripts/kl8_pick5_development.py --csv $(KL8_CSV) --output $(KL8_DEVELOPMENT_OUTPUT) --frozen-periods $(KL8_FROZEN_PERIODS) $(if $(filter 1 true yes,$(KL8_SMOKE)),--smoke,--protocol $(KL8_PROTOCOL))

kl8-pick5-null-smoke: ## 快乐8选5少量全流程随机模拟
	$(RUN) scripts/kl8_pick5_null.py --csv $(KL8_CSV) --output $(KL8_NULL_OUTPUT) --checkpoint-dir $(KL8_NULL_CHECKPOINT) --frozen-periods $(KL8_FROZEN_PERIODS) --iterations $(KL8_NULL_SMOKE_ITERATIONS) --workers $(KL8_NULL_WORKERS) --smoke

kl8-pick5-null-formal: ## 显式运行至少5000次正式null；默认8进程并使用独立检查点
	$(RUN) scripts/kl8_pick5_null.py --csv $(KL8_CSV) --output $(KL8_FORMAL_NULL_OUTPUT) --protocol $(KL8_PROTOCOL) --reference-report $(KL8_DEVELOPMENT_OUTPUT) --checkpoint-dir $(KL8_FORMAL_NULL_CHECKPOINT) --frozen-periods $(KL8_FROZEN_PERIODS) --iterations $(KL8_FORMAL_NULL_ITERATIONS) --workers $(KL8_NULL_WORKERS)

kl8-pick5-predict-today: ## 当前无准入，安全输出空正式候选
	$(RUN) scripts/kl8_pick5_predict_today.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS)

kl8-feature-discovery-v2: ## 仅用1514期开发区执行快乐8探索性特征发现
	$(RUN) scripts/kl8_feature_discovery_v2.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS) --output $(KL8_FEATURE_DISCOVERY_V2_OUTPUT) --n-jobs $(KL8_FEATURE_DISCOVERY_V2_N_JOBS)

digit-model-scoreboard: ## 汇总锁定模型证据并保持无赢家/放弃决策
	$(RUN) scripts/digit_model_scoreboard.py --output-json $(DIGIT_SCOREBOARD_OUTPUT) --output-markdown docs/model_scoreboard.md

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
	@echo "make digit-probability-v5-register  登记v5不可覆盖开发协议"
	@echo "make digit-probability-v5-development  v5隔离开发挑战器"
	@echo "make digit-probability-v5-null-smoke  v5全流程随机模拟冒烟"
	@echo "make kl8-fetch                       抓取快乐8官方原始JSONL"
	@echo "make kl8-fetch-csv                   显式创建快乐8标准CSV"
	@echo "make kl8-pick4-predict-today         快乐8选4安全空正式候选"
	@echo "make kl8-pick4-test-today            快乐8选4等概率测试组合"
	@echo "make kl8-pick4-rank-challenger       Pick4固定排名挑战器"
	@echo "make kl8-pick5-register              登记快乐8选5开发协议"
	@echo "make kl8-pick5-development           运行快乐8选5开发挑战器"
	@echo "make kl8-pick5-null-smoke            快乐8选5随机模拟冒烟"
	@echo "make kl8-pick5-null-formal           快乐8选5正式5000次null（默认8进程）"
	@echo "make kl8-pick5-predict-today         安全输出空正式候选"
	@echo "make kl8-feature-discovery-v2        运行快乐8隔离探索性特征发现"
	@echo "make digit-model-scoreboard          汇总全部锁定模型证据"
	@echo "make digit-learned-ranker-evaluate  Frozen Test 评估"
	@echo "make digit-learned-ranker-daily     生成研究日报"
	@echo "make ci                             运行完整质量闸门"
