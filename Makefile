install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

dev:
	python -m uvicorn app.main:app --reload --port 3009

test:
	python -m pytest -q

lint:
	python -m ruff check app tests

format:
	python -m ruff format app tests

type-check:
	python -m mypy app

check: lint type-check test
