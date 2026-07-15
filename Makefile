.PHONY: api dashboard test lint format dataset train
api:
	uvicorn app.api.main:app --reload
dashboard:
	streamlit run app/dashboard/main.py
test:
	pytest
lint:
	ruff check .
format:
	ruff format .
dataset:
	python scripts/build_dataset.py
train:
	python scripts/train.py
