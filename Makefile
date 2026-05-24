.PHONY: install run test clean help

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

run:  ## Run full pipeline
	python main.py

run-fast:  ## Run pipeline without ablation (faster)
	python main.py --skip-ablation

test:  ## Run tests
	python -m pytest tests/ -v

clean:  ## Clean outputs
	rm -rf outputs/ data/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
