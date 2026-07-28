# 双色球、超级大乐透、快乐8研究工具

.DEFAULT_GOAL := help
RUN ?= uv run --python 3.11 --with-requirements requirements-dev.txt python
OUTPUT_DIR ?= reports

SSQ_CSV ?= data/ssq/official_history.csv
SSQ_RAW_JSONL ?= data/ssq/raw/history.jsonl
SSQ_FETCH_PERIODS ?= 0
SSQ_ENSEMBLE_OUTPUT ?= $(OUTPUT_DIR)/research/ssq_ensemble_v1.json
SSQ_WEIGHT_V2_OUTPUT ?= $(OUTPUT_DIR)/research/ssq_weight_training_v2.json
SSQ_SMALL_COMPOUND_OUTPUT ?= $(OUTPUT_DIR)/retrospective/ssq_small_compound_top5_full_history_v1.json
SSQ_DIVERSIFIED_OUTPUT ?= $(OUTPUT_DIR)/retrospective/ssq_diversified_portfolio_v2_full_history.json
SSQ_B_STATE ?= state/ssq_diversified_portfolio_v2
SSQ_B_KEY ?=
SSQ_D8_STATE ?= state/ssq_8red1blue_v1
SSQ_D8_KEY ?=
SSQ_E_STATE ?= state/ssq_challenger_e_v1
SSQ_E_KEY ?=
SSQ_E_CURRENT ?= $(OUTPUT_DIR)/research/ssq_challenger_e_v1.json
SSQ_E_HISTORY ?= $(OUTPUT_DIR)/retrospective/ssq_challenger_e_v1_full_history.json
SSQ_E2_CURRENT ?= $(OUTPUT_DIR)/research/ssq_challenger_e2_v1.json
SSQ_E2_SELECTION ?= $(OUTPUT_DIR)/retrospective/ssq_challenger_e2_selection_v1.json
SSQ_E2_ENSEMBLE ?= $(OUTPUT_DIR)/retrospective/ssq_ensemble_v1_through_2026085.json

DLT_RAW_JSONL ?= data/dlt/raw/history.jsonl
DLT_CSV ?= data/dlt/official_history.csv
DLT_RECONCILIATION ?= data/dlt/reconciliation.json
DLT_SEARCH_REPORT ?= $(OUTPUT_DIR)/research/dlt_7plus2_search_v1.json
DLT_VALIDATION_REPORT ?= $(OUTPUT_DIR)/retrospective/dlt_7plus2_validation_v1.json
DLT_C5_DIAGNOSTIC_REPORT ?= $(OUTPUT_DIR)/development/dlt_7plus2_c5_diagnostic_v1.json

KL8_CSV ?= data/kl8/kl8.csv
KL8_RAW_JSONL ?= data/kl8/raw/history.jsonl
KL8_FETCH_PERIODS ?= 0
KL8_FROZEN_PERIODS ?= 500
KL8_PICK4_JOINT_STATE ?= state/kl8_pick4_joint_ab_v1
KL8_PICK4_JOINT_KEY ?= $(HOME)/.hermes/secrets/kl8_pick4_joint_ab_v1.key
KL8_PICK4_GENERATE_KEY ?= 0

.PHONY: setup fmt lint test build run ci clean help \
	ssq-fetch ssq-reconcile ssq-evaluate ssq-weight-train-v2 \
	ssq-small-compound-history ssq-diversified-history \
	ssq-b-register ssq-b-snapshot ssq-b-update ssq-b-status \
	ssq-d8-history ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status \
	ssq-e-current ssq-e-history ssq-e2 \
	ssq-e-register ssq-e-snapshot ssq-e-update ssq-e-status \
	dlt-fetch dlt-reconcile dlt-search dlt-validation dlt-c5-diagnostic \
	kl8-fetch kl8-fetch-csv kl8-pick4-predict kl8-pick4-rank \
	kl8-pick4-joint-initialize kl8-pick4-joint-step kl8-pick4-joint-status

setup: ## 创建仅保留三类彩票的运行目录
	$(RUN) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/ssq/raw','data/dlt/raw','data/kl8/raw','reports','logs']]"

fmt: ## 格式化Python代码
	$(RUN) -m black --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black

lint: ## 格式、导入、风格和类型检查
	$(RUN) -m black --check --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black --check-only
	$(RUN) -m flake8 src tests scripts examples
	$(RUN) -m mypy src

test: ## 全量测试且覆盖率不得低于80%
	$(RUN) -m pytest tests -q --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80

build: ## 编译全部保留源码
	$(RUN) -m compileall -q src scripts examples

run: help

# 双色球
ssq-fetch:
	$(RUN) -m scripts.ssq_fetch_history --periods $(SSQ_FETCH_PERIODS) --output-jsonl $(SSQ_RAW_JSONL)
ssq-reconcile:
	$(RUN) -m scripts.ssq_reconcile_history --raw-jsonl $(SSQ_RAW_JSONL) --output-csv $(SSQ_CSV)
ssq-evaluate:
	$(RUN) -m scripts.ssq_ensemble_v1 --csv $(SSQ_CSV) --output $(SSQ_ENSEMBLE_OUTPUT)
ssq-weight-train-v2:
	$(RUN) -m scripts.ssq_weight_training_v2 --csv $(SSQ_CSV) --output $(SSQ_WEIGHT_V2_OUTPUT)
ssq-small-compound-history:
	$(RUN) -m scripts.ssq_small_compound_top5_history_v1 --csv $(SSQ_CSV) --output $(SSQ_SMALL_COMPOUND_OUTPUT)
ssq-diversified-history:
	$(RUN) -m scripts.ssq_diversified_portfolio_v2_history --csv $(SSQ_CSV) --output $(SSQ_DIVERSIFIED_OUTPUT)
ssq-d8-history:
	$(RUN) -m scripts.ssq_8red1blue_v1_history --csv $(SSQ_CSV) --output $(OUTPUT_DIR)/retrospective/ssq_8red1blue_v1_full_history.json
ssq-e-current:
	$(RUN) -m scripts.ssq_challenger_e_v1 --csv $(SSQ_CSV) --ensemble-report $(SSQ_ENSEMBLE_OUTPUT) --output $(SSQ_E_CURRENT)
ssq-e-history:
	$(RUN) -m scripts.ssq_challenger_e_v1_history --csv $(SSQ_CSV) --output $(SSQ_E_HISTORY)
ssq-e2:
	$(RUN) -m scripts.ssq_challenger_e2_selection_v1 --csv $(SSQ_CSV) --ensemble-report $(SSQ_E2_ENSEMBLE) --output $(SSQ_E2_SELECTION) --current-output $(SSQ_E2_CURRENT)

ssq-b-register ssq-b-snapshot ssq-b-update ssq-b-status:
	@test -n "$(SSQ_B_KEY)" || (echo "必须设置SSQ_B_KEY" >&2; exit 2)
	$(RUN) scripts/ssq_diversified_portfolio_v2_prospective.py $(patsubst ssq-b-%,%,$@) --csv $(SSQ_CSV) --state-dir $(SSQ_B_STATE) --hmac-key-file $(SSQ_B_KEY) $(if $(filter register snapshot,$(patsubst ssq-b-%,%,$@)),--ensemble-report $(SSQ_ENSEMBLE_OUTPUT),)

ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status:
	@test -n "$(SSQ_D8_KEY)" || (echo "必须设置SSQ_D8_KEY" >&2; exit 2)
	$(RUN) scripts/ssq_8red1blue_v1_prospective.py $(patsubst ssq-d8-%,%,$@) --csv $(SSQ_CSV) --state-dir $(SSQ_D8_STATE) --hmac-key-file $(SSQ_D8_KEY) $(if $(filter register snapshot,$(patsubst ssq-d8-%,%,$@)),--ensemble-report $(SSQ_ENSEMBLE_OUTPUT),)

ssq-e-register ssq-e-snapshot ssq-e-update ssq-e-status:
	@test -n "$(SSQ_E_KEY)" || (echo "必须设置SSQ_E_KEY" >&2; exit 2)
	@test -n "$(SSQ_D8_KEY)" || (echo "必须设置SSQ_D8_KEY" >&2; exit 2)
	$(RUN) scripts/ssq_challenger_e_v1_prospective.py $(patsubst ssq-e-%,%,$@) --csv $(SSQ_CSV) --state-dir $(SSQ_E_STATE) --hmac-key-file $(SSQ_E_KEY) $(if $(filter register snapshot,$(patsubst ssq-e-%,%,$@)),--e-report $(SSQ_E_CURRENT) --ensemble-report $(SSQ_ENSEMBLE_OUTPUT) --d8-state-dir $(SSQ_D8_STATE) --d8-hmac-key-file $(SSQ_D8_KEY),$(if $(filter update,$(patsubst ssq-e-%,%,$@)),--d8-state-dir $(SSQ_D8_STATE) --d8-hmac-key-file $(SSQ_D8_KEY),))

# 大乐透
dlt-fetch:
	$(RUN) -m scripts.dlt_fetch_history --output-jsonl $(DLT_RAW_JSONL)
dlt-reconcile:
	$(RUN) -m scripts.dlt_reconcile_history --raw-jsonl $(DLT_RAW_JSONL) --output-csv $(DLT_CSV) --output-report $(DLT_RECONCILIATION)
dlt-search:
	$(RUN) -m src.analysis.dlt_7plus2_search_v1 --input-csv $(DLT_CSV) --output-report $(DLT_SEARCH_REPORT)
dlt-validation:
	$(RUN) -m src.analysis.dlt_7plus2_validation_v1 --input-csv $(DLT_CSV) --search-report $(DLT_SEARCH_REPORT) --output-report $(DLT_VALIDATION_REPORT)
dlt-c5-diagnostic:
	$(RUN) scripts/dlt_7plus2_c5_diagnostic_v1.py --csv $(DLT_CSV) --output $(DLT_C5_DIAGNOSTIC_REPORT)

# 快乐8
kl8-fetch:
	$(RUN) scripts/kl8_fetch_history.py --periods $(KL8_FETCH_PERIODS) --output-jsonl $(KL8_RAW_JSONL)
kl8-fetch-csv:
	$(RUN) scripts/kl8_fetch_history.py --periods $(KL8_FETCH_PERIODS) --output-csv $(KL8_CSV)
kl8-pick4-predict:
	$(RUN) scripts/kl8_pick4_predict_today.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS)
kl8-pick4-rank:
	$(RUN) scripts/kl8_pick4_rank_challenger.py --csv $(KL8_CSV) --frozen-periods $(KL8_FROZEN_PERIODS)
kl8-pick4-joint-initialize:
	$(RUN) scripts/kl8_pick4_joint_ab_v1.py initialize --csv $(KL8_CSV) --state-dir $(KL8_PICK4_JOINT_STATE) --hmac-key-file $(KL8_PICK4_JOINT_KEY) $(if $(filter 1 true yes,$(KL8_PICK4_GENERATE_KEY)),--generate-hmac-key,)
kl8-pick4-joint-step:
	$(RUN) scripts/kl8_pick4_joint_ab_v1.py step --csv $(KL8_CSV) --state-dir $(KL8_PICK4_JOINT_STATE) --hmac-key-file $(KL8_PICK4_JOINT_KEY)
kl8-pick4-joint-status:
	$(RUN) scripts/kl8_pick4_joint_ab_v1.py status --csv $(KL8_CSV) --state-dir $(KL8_PICK4_JOINT_STATE) --hmac-key-file $(KL8_PICK4_JOINT_KEY)

ci: lint test build
	@echo "本地 CI 全部通过"

clean:
	$(RUN) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for n in ['build','dist','.pytest_cache','htmlcov'] for p in [pathlib.Path(n)]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

help:
	@echo "保留玩法：双色球(ssq)、超级大乐透(dlt)、快乐8(kl8)"
	@echo "make ssq-evaluate | dlt-search | dlt-validation | kl8-pick4-joint-status"
	@echo "make ci  运行完整质量闸门"
