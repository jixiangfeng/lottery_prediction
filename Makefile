# 双色球 D8 / D8+7 研究工具
.DEFAULT_GOAL := help
RUN ?= uv run --python 3.11 --with-requirements requirements-dev.txt python
OUTPUT_DIR ?= reports
SSQ_CSV ?= data/ssq/official_history.csv
SSQ_RAW_JSONL ?= data/ssq/raw/history.jsonl
SSQ_FETCH_PERIODS ?= 0
SSQ_ENSEMBLE_OUTPUT ?= $(OUTPUT_DIR)/research/ssq_ensemble_v1.json
SSQ_D8_STATE ?= state/ssq_8red1blue_v1
SSQ_D8_KEY ?=

.PHONY: setup fmt lint test build run ci clean help ssq-fetch ssq-reconcile ssq-evaluate ssq-d8-history ssq-d8-official-backtest ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status
setup:
	$(RUN) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/ssq/raw','reports','logs']]"
fmt:
	$(RUN) -m black --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black
lint:
	$(RUN) -m black --check --fast src tests scripts examples
	$(RUN) -m isort src tests scripts examples --profile black --check-only
	$(RUN) -m flake8 src tests scripts examples
	$(RUN) -m mypy src
test:
	$(RUN) -m pytest tests -q --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=0
	$(RUN) -m coverage report --fail-under=80
build:
	$(RUN) -m compileall -q src scripts examples
ci: fmt lint test build
run: help
ssq-fetch:
	$(RUN) -m scripts.ssq_fetch_history --periods $(SSQ_FETCH_PERIODS) --output-jsonl $(SSQ_RAW_JSONL)
ssq-reconcile:
	$(RUN) -m scripts.ssq_reconcile_history --raw-jsonl $(SSQ_RAW_JSONL) --output-csv $(SSQ_CSV)
ssq-evaluate:
	$(RUN) -m scripts.ssq_ensemble_v1 --csv $(SSQ_CSV) --output $(SSQ_ENSEMBLE_OUTPUT)
ssq-d8-history:
	$(RUN) -m scripts.ssq_8red1blue_v1_history --csv $(SSQ_CSV) --output $(OUTPUT_DIR)/retrospective/ssq_8red1blue_v1_full_history.json
ssq-d8-official-backtest:
	$(RUN) -m scripts.ssq_d8_official_backtest --csv $(SSQ_CSV) --prizegrades data/ssq/prizegrades_snapshot.json --output $(OUTPUT_DIR)/retrospective/ssq_d8_official_backtest_v1.json

ssq-d8-register ssq-d8-snapshot ssq-d8-update ssq-d8-status:
	@test -n "$(SSQ_D8_KEY)" || (echo "必须设置SSQ_D8_KEY" >&2; exit 2)
	$(RUN) scripts/ssq_8red1blue_v1_prospective.py $(patsubst ssq-d8-%,%,$@) --csv $(SSQ_CSV) --state-dir $(SSQ_D8_STATE) --hmac-key-file $(SSQ_D8_KEY) $(if $(filter register snapshot,$(patsubst ssq-d8-%,%,$@)),--ensemble-report $(SSQ_ENSEMBLE_OUTPUT),)
clean:
	$(RUN) -c "from pathlib import Path; [p.unlink() for p in Path('reports').rglob('*.json') if p.is_file()]"
help:
	@echo '双色球 D8 / D8+7：make ssq-fetch ssq-reconcile ssq-evaluate ssq-d8-history ssq-d8-official-backtest make ci'
