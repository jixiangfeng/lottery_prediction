# 双色球D8与超级大乐透研究工具

.DEFAULT_GOAL := help
RUN ?= uv run --python 3.11 --with-requirements requirements-dev.txt python
OUTPUT_DIR ?= reports

SSQ_CSV ?= data/ssq/official_history.csv
SSQ_RAW_JSONL ?= data/ssq/raw/history.jsonl
SSQ_FETCH_PERIODS ?= 0
SSQ_ENSEMBLE_OUTPUT ?= $(OUTPUT_DIR)/research/ssq_ensemble_v1.json
SSQ_D8_STATE ?= state/ssq_8red1blue_v1
SSQ_D8_KEY ?=

DLT_RAW_JSONL ?= data/dlt/raw/history.jsonl
DLT_CSV ?= data/dlt/official_history.csv
DLT_RECONCILIATION ?= data/dlt/reconciliation.json
DLT_SEARCH_REPORT ?= $(OUTPUT_DIR)/research/dlt_7plus2_search_v1.json
DLT_VALIDATION_REPORT ?= $(OUTPUT_DIR)/retrospective/dlt_7plus2_validation_v1.json
DLT_C5_DIAGNOSTIC_REPORT ?= $(OUTPUT_DIR)/development/dlt_7plus2_c5_diagnostic_v1.json

.PHONY: setup fmt lint test build run ci clean help \
	ssq-fetch ssq-reconcile ssq-evaluate ssq-d8-history \
	ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status \
	dlt-fetch dlt-reconcile dlt-search dlt-validation dlt-c5-diagnostic

setup: ## 创建运行目录
	$(RUN) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/ssq/raw','data/dlt/raw','reports','logs']]"

fmt: ## 格式化Python代码
	$(RUN) -m black --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black

lint: ## 格式、导入、风格和类型检查
	$(RUN) -m black --check --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black --check-only
	$(RUN) -m flake8 src tests scripts examples
	$(RUN) -m mypy src

test: ## 全量测试且覆盖率不得低于80%
	$(RUN) -m pytest tests -q --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=0
	$(RUN) -m coverage report --fail-under=80

build: ## 编译全部保留源码
	$(RUN) -m compileall -q src scripts examples

run: help

# 双色球：保留D8、D8+7单式与Top20分层v3.3.1构造
ssq-fetch:
	$(RUN) -m scripts.ssq_fetch_history --periods $(SSQ_FETCH_PERIODS) --output-jsonl $(SSQ_RAW_JSONL)
ssq-reconcile:
	$(RUN) -m scripts.ssq_reconcile_history --raw-jsonl $(SSQ_RAW_JSONL) --output-csv $(SSQ_CSV)
ssq-evaluate:
	$(RUN) -m scripts.ssq_ensemble_v1 --csv $(SSQ_CSV) --output $(SSQ_ENSEMBLE_OUTPUT)
ssq-d8-history:
	$(RUN) -m scripts.ssq_8red1blue_v1_history --csv $(SSQ_CSV) --output $(OUTPUT_DIR)/retrospective/ssq_8red1blue_v1_full_history.json

ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status:
	@test -n "$(SSQ_D8_KEY)" || (echo "必须设置SSQ_D8_KEY" >&2; exit 2)
	$(RUN) scripts/ssq_8red1blue_v1_prospective.py $(patsubst ssq-d8-%,%,$@) --csv $(SSQ_CSV) --state-dir $(SSQ_D8_STATE) --hmac-key-file $(SSQ_D8_KEY) $(if $(filter register snapshot,$(patsubst ssq-d8-%,%,$@)),--ensemble-report $(SSQ_ENSEMBLE_OUTPUT),)

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

ci: lint test build
	@echo "本地 CI 全部通过"

clean:
	$(RUN) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for n in ['build','dist','.pytest_cache','htmlcov'] for p in [pathlib.Path(n)]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

help:
	@echo "双色球仅保留D8、D8+7单式、Top20分层v3.3.1；保留超级大乐透(dlt)"
	@echo "make ssq-evaluate | ssq-d8-history | dlt-search | dlt-validation"
	@echo "make ci  运行完整质量闸门"
